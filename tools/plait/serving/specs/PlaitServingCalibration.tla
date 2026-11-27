---- MODULE PlaitServingCalibration ----
\* A projected P/D layout may reserve two candidates. Its private benchmark
\* keeps one immutable binding per role. Config and active changes can race
\* with the job, which must validate the context before caching a result.
\* Time, throughput and the earlier placement policy are abstracted. The
\* policy revision represents the calibration settings. Native operations use
\* an ordered active view; HTTP probes are local observations.
EXTENDS Naturals

Aggregate == <<1, 1>>
Prefill == 2
Decode == 3
Roles == {Prefill, Decode}
Spare == <<4, 1>>
Incumbent == {Aggregate}
Epochs == 2..3
Revisions == 1..2
Origin(role, epoch, route) == 100 * role + 10 * epoch + route
Binding(role, epoch, origin) == [version |-> <<role, epoch>>, origin |-> origin]
BindingSet(epoch, route) == {Binding(role, epoch, Origin(role, epoch, route)) : role \in Roles}
Versions(bindings) == {b.version : b \in bindings}
AllBindings == UNION {BindingSet(epoch, route) : epoch \in Epochs, route \in Revisions}
Context(route, policy, active, pins) == [route |-> route, policy |-> policy, active |-> active, pins |-> pins]
NoView == Context(0, 0, {}, {})
Job(phase, view, ran) == [phase |-> phase, view |-> view, ran |-> ran]
Result(grade, valid, pins) == [grade |-> grade, valid |-> valid, pins |-> pins]
Profile(valid, route, policy, pins, used) ==
    [valid |-> valid, route |-> route, policy |-> policy, pins |-> pins, used |-> used]
Switch(phase, expected, target) == [phase |-> phase, expected |-> expected, target |-> target]

VARIABLES native, config, job, result, cache, load, cutover
vars == <<native, config, job, result, cache, load, cutover>>

Init ==
    /\ native = [active |-> Incumbent, epoch |-> 0, ready |-> {}, health |-> {},
                 lost |-> {}, disturbed |-> FALSE]
    /\ config = [route |-> 1, policy |-> 1, projected |-> TRUE, incumbentMeasured |-> TRUE]
    /\ job = Job("Idle", NoView, {})
    /\ result = Result("None", FALSE, {})
    /\ cache = Profile(FALSE, 0, 0, {}, FALSE)
    /\ load = [arrivals |-> 0, public |-> 0, private |-> 0]
    /\ cutover = Switch("Idle", {}, {})

Pins == IF native.epoch = 0 THEN {} ELSE BindingSet(native.epoch, config.route)
View == Context(config.route, config.policy, native.active, Pins)
CompleteCandidate == native.epoch \in Epochs /\ native.ready = Roles
PrivateCandidate == Versions(Pins) \cap native.active = {}

Prepare ==
    /\ native.epoch = 0 /\ config.projected /\ config.incumbentMeasured
    /\ native' = [native EXCEPT !.epoch = 2]
    /\ UNCHANGED <<config, job, result, cache, load, cutover>>

ReplaceCandidate ==
    /\ native.epoch = 2 /\ cutover.phase = "Idle" /\ PrivateCandidate
    /\ native' = [native EXCEPT !.epoch = 3, !.ready = {}, !.health = {}, !.lost = {}]
    /\ UNCHANGED <<config, job, result, cache, load, cutover>>

SetReady(role) ==
    /\ native.epoch \in Epochs /\ role \notin native.ready \cup native.lost
    /\ native' = [native EXCEPT !.ready = @ \cup {role}]
    /\ UNCHANGED <<config, job, result, cache, load, cutover>>

ObserveHealth(role) ==
    /\ role \in native.ready \ native.health
    /\ native' = [native EXCEPT !.health = @ \cup {role}]
    /\ UNCHANGED <<config, job, result, cache, load, cutover>>

LoseHttp(role) ==
    /\ role \in native.health
    /\ native' = [native EXCEPT !.health = @ \ {role}]
    /\ UNCHANGED <<config, job, result, cache, load, cutover>>

LoseReady(role) ==
    /\ role \in native.ready
    /\ native' = [native EXCEPT !.ready = @ \ {role}, !.health = @ \ {role},
                 !.lost = @ \cup {role}, !.active = @ \ {<<role, native.epoch>>}]
    /\ UNCHANGED <<config, job, result, cache, load, cutover>>

ChangeRoute ==
    /\ config.route = 1
    /\ config' = [config EXCEPT !.route = 2]
    /\ UNCHANGED <<native, job, result, cache, load, cutover>>

ChangePolicy ==
    /\ config.policy = 1
    /\ config' = [config EXCEPT !.policy = 2]
    /\ UNCHANGED <<native, job, result, cache, load, cutover>>

ExpireIncumbent ==
    /\ config.incumbentMeasured
    /\ config' = [config EXCEPT !.incumbentMeasured = FALSE]
    /\ UNCHANGED <<native, job, result, cache, load, cutover>>

ExpireProjection ==
    /\ config.projected
    /\ config' = [config EXCEPT !.projected = FALSE]
    /\ UNCHANGED <<native, job, result, cache, load, cutover>>

IndependentActiveChange(versions) ==
    /\ ~native.disturbed /\ versions # native.active
    /\ native' = [native EXCEPT !.active = versions, !.disturbed = TRUE]
    /\ UNCHANGED <<config, job, result, cache, load, cutover>>

StartBenchmark ==
    /\ job.phase = "Idle" /\ CompleteCandidate /\ PrivateCandidate /\ native.health = Roles
    /\ config.projected /\ config.incumbentMeasured
    /\ job' = Job("Running", View, {})
    /\ UNCHANGED <<native, config, result, cache, load, cutover>>

\* The private ingress checks the pinned view at dispatch. A probe already
\* dispatched may finish after a change, and its result can arrive late.
RunProbe(binding) ==
    /\ job.phase = "Running" /\ job.view = View /\ binding \in job.view.pins \ job.ran
    /\ job' = [job EXCEPT !.ran = @ \cup {binding}]
    /\ load' = [load EXCEPT !.private = @ + 1]
    /\ UNCHANGED <<native, config, result, cache, cutover>>

UserArrival ==
    /\ load.public = 0
    /\ load' = [load EXCEPT !.public = @ + 1, !.arrivals = @ + 1]
    /\ UNCHANGED <<native, config, job, result, cache, cutover>>

WrongOrigin(pins) == {Binding(b.version[1], b.version[2], b.origin + 1) : b \in pins}

\* The pins on a result record describe its provenance in this model. The
\* benchmark output need not carry an origin field on the wire.
Finish(grade, exact) ==
    /\ job.phase \in {"Running", "Cancelled"} /\ job.ran = job.view.pins /\ result.grade = "None"
    /\ result' = Result(grade, TRUE, IF exact THEN job.view.pins ELSE WrongOrigin(job.view.pins))
    /\ job' = IF job.phase = "Running" THEN [job EXCEPT !.phase = "Returned"] ELSE job
    /\ UNCHANGED <<native, config, cache, load, cutover>>

Cancel(cause) ==
    /\ job.phase \in {"Running", "Returned"}
    /\ cause # "Changed" \/ job.view # View \/ ~CompleteCandidate \/ ~PrivateCandidate
    /\ job' = [job EXCEPT !.phase = "Cancelled"]
    /\ UNCHANGED <<native, config, result, cache, load, cutover>>

ExpireResult ==
    /\ result.valid
    /\ result' = [result EXCEPT !.valid = FALSE]
    /\ UNCHANGED <<native, config, job, cache, load, cutover>>

Acceptable ==
    /\ job.phase = "Returned" /\ job.view = View /\ CompleteCandidate /\ PrivateCandidate /\ native.health = Roles
    /\ result.grade = "Positive" /\ result.valid /\ result.pins = job.view.pins

StoreResult ==
    /\ Acceptable
    /\ cache' = Profile(TRUE, config.route, config.policy, result.pins, FALSE)
    /\ job' = [job EXCEPT !.phase = "Stored"]
    /\ UNCHANGED <<native, config, result, load, cutover>>

RejectResult ==
    /\ job.phase = "Returned" /\ ~Acceptable
    /\ job' = [job EXCEPT !.phase = "Cancelled"]
    /\ UNCHANGED <<native, config, result, cache, load, cutover>>

ExpireCache ==
    /\ cache.valid
    /\ cache' = [cache EXCEPT !.valid = FALSE]
    /\ UNCHANGED <<native, config, job, result, load, cutover>>

MeasuredCurrent ==
    cache.valid /\ cache.route = config.route /\ cache.policy = config.policy /\ cache.pins = Pins

ObserveToSwitch ==
    /\ native.epoch \in Epochs /\ cutover.phase = "Idle" /\ cutover.expected # native.active
    /\ cutover' = [cutover EXCEPT !.expected = native.active]
    /\ UNCHANGED <<native, config, job, result, cache, load>>

SendSwitch ==
    /\ cutover.phase = "Idle" /\ cutover.expected = Incumbent
    /\ config.incumbentMeasured /\ MeasuredCurrent /\ CompleteCandidate /\ native.health = Roles
    /\ cutover' = Switch("Sent", cutover.expected, Versions(cache.pins))
    /\ UNCHANGED <<native, config, job, result, cache, load>>

ApplySwitch ==
    /\ cutover.phase = "Sent"
    /\ LET match == native.active = cutover.expected
                      /\ cutover.target = Versions(Pins) /\ CompleteCandidate
       IN /\ native' = IF match THEN [native EXCEPT !.active = cutover.target] ELSE native
          /\ cutover' = [cutover EXCEPT !.phase = IF match THEN "Applied" ELSE "Rejected"]
    /\ UNCHANGED <<config, job, result, cache, load>>

\* A measured profile is keyed by layout, origins and config revisions. It
\* can describe the incumbent after cutover even though the active set moved.
ReadBaseline ==
    /\ cutover.phase = "Applied" /\ MeasuredCurrent /\ native.active = Versions(cache.pins)
    /\ cache' = [cache EXCEPT !.used = TRUE]
    /\ UNCHANGED <<native, config, job, result, load, cutover>>

Next ==
    \/ Prepare \/ ReplaceCandidate \/ ChangeRoute \/ ChangePolicy \/ ExpireIncumbent \/ ExpireProjection
    \/ \E role \in Roles : SetReady(role) \/ ObserveHealth(role) \/ LoseHttp(role) \/ LoseReady(role)
    \/ IndependentActiveChange(native.active \cup {Spare})
    \/ Aggregate \in native.active /\ IndependentActiveChange(native.active \ {Aggregate})
    \/ CompleteCandidate /\ IndependentActiveChange(native.active \cup {<<Prefill, native.epoch>>})
    \/ StartBenchmark \/ \E binding \in AllBindings : RunProbe(binding)
    \/ UserArrival \/ \E grade \in {"Positive", "NoThroughput"}, exact \in BOOLEAN : Finish(grade, exact)
    \/ \E cause \in {"Timeout", "Failed", "Changed"} : Cancel(cause)
    \/ ExpireResult \/ StoreResult \/ RejectResult \/ ExpireCache
    \/ ObserveToSwitch \/ SendSwitch \/ ApplySwitch \/ ReadBaseline

Spec == Init /\ [][Next]_vars

ProjectedPreparationOnly ==
    ~ENABLED (Prepare /\ (~config.projected \/ ~config.incumbentMeasured \/ native'.active # native.active))

PrivateCompleteAtStart ==
    ~ENABLED (StartBenchmark /\
        (native.epoch = 0 \/ native.ready # Roles \/ native.health # Roles \/
         Versions(Pins) \cap native.active # {} \/ job'.view # View))

BenchmarkUsesPins ==
    /\ job.ran \subseteq job.view.pins
    /\ \A binding \in AllBindings :
          ~ENABLED (RunProbe(binding) /\
              (job.view # View \/ binding \notin job.view.pins \/ job'.view # job.view))

BackgroundLoadExcluded == load.arrivals = load.public

OnlyCurrentPositiveResults ==
    ~ENABLED (StoreResult /\
        (job.phase # "Returned" \/ job.view.route # config.route \/ job.view.policy # config.policy \/
         job.view.active # native.active \/ job.view.pins # Pins \/
         native.ready # Roles \/ native.health # Roles \/ Versions(Pins) \cap native.active # {} \/
         ~result.valid \/ result.grade # "Positive" \/ result.pins # job.view.pins))

CachedProfileKeepsExactPins ==
    cache.route # 0 =>
        /\ job.phase = "Stored" /\ cache.route = job.view.route /\ cache.policy = job.view.policy
        /\ cache.pins = job.view.pins /\ result.grade = "Positive"

SendNeedsMeasuredHealthyLayout ==
    ~ENABLED (SendSwitch /\
        (~cache.valid \/ cache.route # config.route \/ cache.policy # config.policy \/ cache.pins # Pins \/
         ~config.incumbentMeasured \/ native.ready # Roles \/ native.health # Roles \/
         cutover'.target # Versions(cache.pins)))

OrderedCutoverComparesActive ==
    cutover.phase = "Sent" =>
        LET legal == native.active = cutover.expected /\ cutover.target = Versions(Pins) /\ native.ready = Roles
        IN /\ ENABLED ApplySwitch
           /\ IF legal THEN ~ENABLED (ApplySwitch /\ (cutover'.phase # "Applied" \/ native'.active # cutover.target))
              ELSE ~ENABLED (ApplySwitch /\ (cutover'.phase # "Rejected" \/ native' # native))

BaselineReusesMeasuredLayout ==
    /\ ~ENABLED (ReadBaseline /\ (~cache.valid \/ cache.route # config.route \/ cache.policy # config.policy \/
            cache.pins # Pins \/ native.active # Versions(cache.pins)))
    /\ cutover.phase = "Applied" /\ MeasuredCurrent /\ native.active = Versions(cache.pins) =>
           ENABLED (ReadBaseline /\ cache'.used)
====
