---- MODULE PlaitMeasuredDeviceReuse ----
EXTENDS Naturals

CONSTANTS CheckProfile, CheckVersion, CheckLifetime, CheckCapacity,
          KeepWarmupProfiles, CancelWarmupRun, CheckFirstTokens, CheckTimedEvidence
Candidates == 1..2
Streams == 1..2
WarmupPhases == {"Warmup", "Injected"}
NoRun == [candidate |-> 0, epoch |-> 0, phase |-> "Idle", first |-> {}]
NoProfile == [epoch |-> 0, fresh |-> FALSE, covers |-> FALSE]

VARIABLES active, versions, run, profiles, evidence, warmups
vars == <<active, versions, run, profiles, evidence, warmups>>

Init ==
    /\ active = 0
    /\ versions = [candidate \in Candidates |-> 1]
    /\ warmups = 2
    /\ run = NoRun
    /\ profiles = [candidate \in Candidates |-> NoProfile]
    /\ evidence = [measured |-> TRUE, exact |-> TRUE, fresh |-> TRUE, covers |-> TRUE]

\* Candidate 1 is the ready bridge. Candidate 2 is the ready replacement.
\* The resource forecast and lease return are covered by the reuse model.
ChangeEpoch(candidate) ==
    /\ active < candidate /\ versions[candidate] = 1
    /\ versions' = [versions EXCEPT ![candidate] = 2]
    /\ UNCHANGED <<active, run, profiles, evidence, warmups>>

StartBenchmark(withWarmup) ==
    /\ active < 2 /\ run = NoRun
    /\ run' = [candidate |-> active + 1, epoch |-> versions[active + 1],
               phase |-> IF withWarmup THEN "Warmup" ELSE "Timed", first |-> {}]
    /\ UNCHANGED <<active, versions, profiles, evidence, warmups>>

FirstToken(stream) ==
    /\ run.phase = "Warmup" /\ stream \notin run.first
    /\ run' = [run EXCEPT !.first = @ \cup {stream}]
    /\ UNCHANGED <<active, versions, profiles, evidence, warmups>>

InjectWarmup ==
    /\ run.phase = "Warmup"
    /\ (~CheckFirstTokens \/ run.first = Streams)
    /\ run' = [run EXCEPT !.phase = "Injected"]
    /\ UNCHANGED <<active, versions, profiles, evidence, warmups>>

FinishWarmup ==
    /\ run.phase = "Injected"
    /\ run' = [run EXCEPT !.phase = "Timed"]
    /\ UNCHANGED <<active, versions, profiles, evidence, warmups>>

FinishBenchmark(sufficient) ==
    /\ run # NoRun
    /\ (~CheckTimedEvidence \/ run.phase = "Timed")
    /\ profiles' = [profiles EXCEPT ![run.candidate] =
                    [epoch |-> run.epoch, fresh |-> TRUE, covers |-> sufficient]]
    /\ run' = NoRun
    /\ UNCHANGED <<active, versions, evidence, warmups>>

\* Warmup count changes the next job, while completed measurements on
\* the same version remain evidence for the same timed workload.
TuneWarmups ==
    /\ warmups = 2
    /\ warmups' = 8
    /\ profiles' = IF KeepWarmupProfiles THEN profiles
                   ELSE [candidate \in Candidates |-> NoProfile]
    /\ run' = IF CancelWarmupRun THEN NoRun ELSE run
    /\ UNCHANGED <<active, versions, evidence>>

Expire(candidate) ==
    /\ profiles[candidate].fresh
    /\ profiles' = [profiles EXCEPT ![candidate].fresh = FALSE]
    /\ UNCHANGED <<active, versions, run, evidence, warmups>>

Switch ==
    /\ active < 2
    /\ LET candidate == active + 1
           profile == profiles[candidate]
           measured == profile.epoch # 0
           exact == profile.epoch = versions[candidate]
       IN /\ IF measured
             THEN /\ (~CheckVersion \/ exact)
                  /\ (~CheckLifetime \/ profile.fresh)
                  /\ (~CheckCapacity \/ profile.covers)
             ELSE ~CheckProfile
          /\ evidence' = [measured |-> measured, exact |-> exact,
                          fresh |-> profile.fresh, covers |-> profile.covers]
          /\ active' = candidate
    /\ UNCHANGED <<versions, run, profiles, warmups>>

Next ==
    \/ \E withWarmup \in BOOLEAN : StartBenchmark(withWarmup)
    \/ \E stream \in Streams : FirstToken(stream)
    \/ InjectWarmup \/ FinishWarmup \/ Switch \/ TuneWarmups
    \/ \E candidate \in Candidates : ChangeEpoch(candidate) \/ Expire(candidate)
    \/ \E sufficient \in BOOLEAN : FinishBenchmark(sufficient)

Spec == Init /\ [][Next]_vars
RequiresMeasurement == evidence.measured
NeedsCurrentEpoch == evidence.exact
NeedsFreshProfile == evidence.fresh
CoversArrivalRate == evidence.covers
WarmupPreservesCompletedProfiles == [][warmups' # warmups => profiles' = profiles]_vars
WarmupCancelsPendingBenchmark == [][warmups' # warmups => run' = NoRun]_vars
FirstTokensBeforeInjection == [][run.phase = "Warmup" /\ run'.phase = "Injected" =>
                                 run.first = Streams]_vars
WarmupCannotPublishProfile ==
    [][run.phase \in WarmupPhases /\ run'.phase \notin WarmupPhases =>
       profiles' = profiles]_vars
====
