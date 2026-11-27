---- MODULE PlaitServingProvision ----
\* A measured incumbent can provision either a P/D pair or a new aggregate.
\* Two complete load windows stand for the policy's stable projected gain.
\* All native operations are ordered. The native prepare uses one reservation
\* budget and records one receipt. Launch authorization is visible before the
\* controller receives that receipt. Candidate generations and IDs are bounded.
\* The measured switch policy's load evidence is abstracted at this boundary.
\* HTTP and profile checks occur before sending the switch. At execution the
\* native operation rechecks active epochs and candidate group readiness.
EXTENDS Naturals, Sequences

Aggregate == <<1, 1>>
Prefill == <<2, 2>>
Decode == <<3, 2>>
Replacement == <<4, 2>>
Spare == <<5, 1>>
Groups == 1..5
Incumbent == {Aggregate}
Split == {Prefill, Decode}
FutureAggregate == {Replacement}
Layouts == {Split, FutureAggregate}
Candidates == Split \cup FutureAggregate
NativeReady == {"Communicator", "Http"}
Pending == "Pending"
Accepted == "Accepted"
Rejected == "Rejected"

VersionsFor(versions, group) == {v \in versions : v[1] = group}
SameActiveSet(versions, observed) ==
    \A g \in Groups : VersionsFor(versions, g) = VersionsFor(observed, g)

Evidence(observed, target, revision) ==
    [active |-> observed, target |-> target, revision |-> revision]
PrepareIntent(id, expected, target) ==
    [id |-> id, expected |-> expected, target |-> target, outcome |-> Pending, reply |-> Pending, retried |-> FALSE]
SwitchIntent(id, expected) == [id |-> id, expected |-> expected, outcome |-> Pending]
Profile(versions, valid) == [versions |-> versions, valid |-> valid]

VARIABLES native, policy, prepare, switch, profiles
vars == <<native, policy, prepare, switch, profiles>>

Init ==
    /\ native = [active |-> Incumbent, reserved |-> {}, enough |-> TRUE,
                 phase |-> [v \in Candidates |-> "Absent"], disturbed |-> FALSE]
    /\ policy = [revision |-> 1, observed |-> {}, unread |-> FALSE, full |-> FALSE, support |-> <<>>]
    /\ prepare = PrepareIntent(0, {}, {})
    /\ switch = SwitchIntent(0, {})
    /\ profiles = [incumbent |-> TRUE, projected |-> Layouts, measured |-> Profile({}, FALSE)]

ObservePrepare ==
    /\ prepare.id = 0 /\ ~policy.unread
    /\ policy' = [policy EXCEPT !.observed = native.active, !.unread = TRUE]
    /\ UNCHANGED <<native, prepare, switch, profiles>>

FullInterval ==
    /\ prepare.id = 0 /\ ~policy.full
    /\ policy' = [policy EXCEPT !.full = TRUE]
    /\ UNCHANGED <<native, prepare, switch, profiles>>

ShortPoll ==
    /\ prepare.id = 0 /\ policy.unread /\ ~policy.full
    /\ policy' = [policy EXCEPT !.unread = FALSE]
    /\ UNCHANGED <<native, prepare, switch, profiles>>

Reload ==
    /\ prepare.id = 0 /\ policy.revision = 1
    /\ policy' = [policy EXCEPT !.revision = 2, !.unread = FALSE, !.full = FALSE, !.support = <<>>]
    /\ UNCHANGED <<native, prepare, switch, profiles>>

SupportsPrepare(target) ==
    target \in profiles.projected /\ profiles.incumbent /\ policy.observed = Incumbent

ExtendEvidence(evidence) ==
    LET n == Len(policy.support) IN
    IF n = 0 THEN <<evidence>>
    ELSE IF policy.support[n] = evidence THEN <<policy.support[n], evidence>> ELSE <<evidence>>

AssessWindow(target) ==
    /\ prepare.id = 0 /\ policy.unread /\ policy.full
    /\ LET evidence == Evidence(policy.observed, target, policy.revision)
       IN policy' = [policy EXCEPT !.unread = FALSE, !.full = FALSE,
                     !.support = IF SupportsPrepare(target) THEN ExtendEvidence(evidence) ELSE <<>>]
    /\ UNCHANGED <<native, prepare, switch, profiles>>

SendPrepare ==
    /\ prepare.id = 0 /\ Len(policy.support) = 2
    /\ LET evidence == policy.support[2]
       IN /\ profiles.incumbent /\ evidence.target \in profiles.projected
          /\ prepare' = PrepareIntent(1, evidence.active, evidence.target)
    /\ UNCHANGED <<native, policy, switch, profiles>>

ApplyPrepare ==
    /\ prepare.id = 1 /\ prepare.outcome = Pending
    /\ LET accepted == native.active = prepare.expected /\ native.enough
       IN /\ native' = IF accepted
              THEN [native EXCEPT !.reserved = prepare.target,
                    !.phase = [v \in Candidates |-> IF v \in prepare.target THEN "Granted" ELSE "Absent"]]
              ELSE native
          /\ prepare' = [prepare EXCEPT !.outcome = IF accepted THEN Accepted ELSE Rejected]
    /\ UNCHANGED <<policy, switch, profiles>>

ReceivePrepare ==
    /\ prepare.outcome # Pending /\ prepare.reply = Pending
    /\ prepare' = [prepare EXCEPT !.reply = prepare.outcome]
    /\ UNCHANGED <<native, policy, switch, profiles>>

\* A dropped reply does not roll back the grant. Retrying the same exact
\* command ID consults the original receipt without preparing again.
RetryPrepare ==
    /\ prepare.outcome # Pending /\ ~prepare.retried
    /\ prepare' = [prepare EXCEPT !.reply = prepare.outcome, !.retried = TRUE]
    /\ UNCHANGED <<native, policy, switch, profiles>>

Advance(v, from, to) ==
    /\ native.phase[v] = from
    /\ native' = [native EXCEPT !.phase[v] = to]
    /\ UNCHANGED <<policy, prepare, switch, profiles>>

Launch(v) == Advance(v, "Granted", "Launched")
CommunicatorReady(v) == Advance(v, "Launched", "Communicator")
HttpReady(v) == Advance(v, "Communicator", "Http")
LoseHttp(v) == Advance(v, "Http", "Communicator")

PublishMeasured(exact) ==
    /\ prepare.outcome = Accepted /\ ~profiles.measured.valid
    /\ LET versions == IF exact THEN prepare.target ELSE {<<v[1], 1>> : v \in prepare.target}
       IN profiles' = [profiles EXCEPT !.measured = Profile(versions, TRUE)]
    /\ UNCHANGED <<native, policy, prepare, switch>>

ExpireMeasured ==
    /\ profiles.measured.valid
    /\ profiles' = [profiles EXCEPT !.measured.valid = FALSE]
    /\ UNCHANGED <<native, policy, prepare, switch>>

ExpireIncumbent ==
    /\ profiles.incumbent
    /\ profiles' = [profiles EXCEPT !.incumbent = FALSE]
    /\ UNCHANGED <<native, policy, prepare, switch>>

ExpireProjection ==
    /\ profiles.projected = Layouts
    /\ profiles' = [profiles EXCEPT !.projected = {}]
    /\ UNCHANGED <<native, policy, prepare, switch>>

ObserveSwitch ==
    /\ prepare.outcome = Accepted /\ switch.id = 0 /\ switch.expected # native.active
    /\ switch' = [switch EXCEPT !.expected = native.active]
    /\ UNCHANGED <<native, policy, prepare, profiles>>

SendSwitch ==
    /\ prepare.outcome = Accepted /\ switch.id = 0 /\ switch.expected = Incumbent
    /\ profiles.incumbent /\ profiles.measured.valid /\ profiles.measured.versions = prepare.target
    /\ \A v \in prepare.target : native.phase[v] = "Http"
    /\ switch' = [switch EXCEPT !.id = 2]
    /\ UNCHANGED <<native, policy, prepare, profiles>>

ApplySwitch ==
    /\ switch.id = 2 /\ switch.outcome = Pending
    /\ LET accepted == native.active = switch.expected /\
                       (\A v \in prepare.target : native.phase[v] \in NativeReady)
       IN /\ native' = IF accepted THEN [native EXCEPT !.active = prepare.target] ELSE native
          /\ switch' = [switch EXCEPT !.outcome = IF accepted THEN Accepted ELSE Rejected]
    /\ UNCHANGED <<policy, prepare, profiles>>

ResourceLoss ==
    /\ prepare.outcome = Pending /\ native.enough
    /\ native' = [native EXCEPT !.enough = FALSE]
    /\ UNCHANGED <<policy, prepare, switch, profiles>>

ActivateSpare ==
    /\ ~native.disturbed
    /\ native' = [native EXCEPT !.active = @ \cup {Spare}, !.disturbed = TRUE]
    /\ UNCHANGED <<policy, prepare, switch, profiles>>

RevokeIncumbent ==
    /\ ~native.disturbed /\ Aggregate \in native.active
    /\ native' = [native EXCEPT !.active = @ \ {Aggregate}, !.disturbed = TRUE]
    /\ UNCHANGED <<policy, prepare, switch, profiles>>

RevokeCandidate(v) ==
    /\ ~native.disturbed /\ v \in native.reserved /\ native.phase[v] # "Revoked"
    /\ native' = [native EXCEPT !.active = @ \ {v}, !.phase[v] = "Revoked", !.disturbed = TRUE]
    /\ UNCHANGED <<policy, prepare, switch, profiles>>

Next ==
    \/ ObservePrepare \/ FullInterval \/ ShortPoll \/ Reload
    \/ \E target \in Layouts \cup {{}} : AssessWindow(target)
    \/ SendPrepare \/ ApplyPrepare \/ ReceivePrepare \/ RetryPrepare
    \/ \E v \in Candidates : Launch(v) \/ CommunicatorReady(v) \/ HttpReady(v) \/ LoseHttp(v) \/ RevokeCandidate(v)
    \/ \E exact \in BOOLEAN : PublishMeasured(exact)
    \/ ExpireMeasured \/ ExpireIncumbent \/ ExpireProjection
    \/ ObserveSwitch \/ SendSwitch \/ ApplySwitch
    \/ ResourceLoss \/ ActivateSpare \/ RevokeIncumbent

Spec == Init /\ [][Next]_vars

StablePrepareEvidence ==
    /\ Len(policy.support) <= 2
    /\ \A i \in DOMAIN policy.support :
          /\ policy.support[i] = policy.support[1]
          /\ policy.support[i].active = Incumbent /\ policy.support[i].target \in Layouts
          /\ policy.support[i].revision = policy.revision
    /\ prepare.id # 0 =>
          /\ Len(policy.support) = 2
          /\ prepare.expected = policy.support[2].active /\ prepare.target = policy.support[2].target
    /\ \A target \in Layouts \cup {{}} :
          ~ENABLED (AssessWindow(target) /\
              (~policy.full \/ policy'.full \/ ~policy.unread \/ policy'.unread))

ObservationsUseWholeSet ==
    /\ ~ENABLED (ObservePrepare /\ policy'.observed # native.active)
    /\ ~ENABLED (ObserveSwitch /\ switch'.expected # native.active)

ProjectionOnlyPrepares ==
    /\ ~ENABLED (SendPrepare /\ (~profiles.incumbent \/ prepare'.target \notin profiles.projected))
    /\ ~ENABLED (SendSwitch /\
            (~profiles.incumbent \/ ~profiles.measured.valid \/ profiles.measured.versions # prepare.target))

PrepareIsAtomic ==
    /\ ~ENABLED (ApplyPrepare /\ native'.active # native.active)
    /\ prepare.id = 1 /\ prepare.outcome = Pending =>
          /\ ENABLED ApplyPrepare
          /\ IF SameActiveSet(native.active, prepare.expected) /\ native.enough THEN
                ~ENABLED (ApplyPrepare /\ (prepare'.outcome # Accepted \/ native'.reserved # prepare.target \/
                              \E v \in prepare.target : native'.phase[v] # "Granted"))
             ELSE ~ENABLED (ApplyPrepare /\ (prepare'.outcome # Rejected \/ native' # native))

LaunchUsesGrant ==
    \A v \in Candidates :
          /\ (native.phase[v] # "Absent") = (v \in native.reserved)
          /\ ~ENABLED (Launch(v) /\ (native.phase[v] # "Granted" \/ v \notin native.reserved))

RetryReturnsReceipt ==
    /\ prepare.reply # Pending => prepare.reply = prepare.outcome
    /\ ~ENABLED (RetryPrepare /\ (native' # native \/ prepare'.id # prepare.id \/
            prepare'.expected # prepare.expected \/ prepare'.target # prepare.target \/
            prepare'.outcome # prepare.outcome \/ prepare'.reply # prepare.outcome))

DispatchNeedsWholeTarget ==
    ~ENABLED (SendSwitch /\ (switch.expected # Incumbent \/
                \E v \in prepare.target : native.phase[v] # "Http"))

SwitchUsesCurrentEpochs ==
    switch.id = 2 /\ switch.outcome = Pending =>
        /\ ENABLED ApplySwitch
        /\ IF SameActiveSet(native.active, switch.expected) /\
              (\A v \in prepare.target : native.phase[v] \in NativeReady) THEN
              ~ENABLED (ApplySwitch /\ (switch'.outcome # Accepted \/ native'.active # prepare.target))
           ELSE ~ENABLED (ApplySwitch /\ (switch'.outcome # Rejected \/ native' # native))
====
