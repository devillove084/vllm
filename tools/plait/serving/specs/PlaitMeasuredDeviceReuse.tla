---- MODULE PlaitMeasuredDeviceReuse ----
EXTENDS Naturals

CONSTANTS CheckProfile, CheckVersion, CheckLifetime, CheckCapacity
Candidates == 1..2
NoRun == [candidate |-> 0, epoch |-> 0]
NoProfile == [epoch |-> 0, fresh |-> FALSE, covers |-> FALSE]

VARIABLES active, versions, run, profiles, evidence
vars == <<active, versions, run, profiles, evidence>>

Init ==
    /\ active = 0
    /\ versions = [candidate \in Candidates |-> 1]
    /\ run = NoRun
    /\ profiles = [candidate \in Candidates |-> NoProfile]
    /\ evidence = [measured |-> TRUE, exact |-> TRUE, fresh |-> TRUE, covers |-> TRUE]

\* Candidate 1 is the ready bridge. Candidate 2 is the ready replacement.
\* The resource forecast and lease return are covered by the reuse model.
ChangeEpoch(candidate) ==
    /\ active < candidate /\ versions[candidate] = 1
    /\ versions' = [versions EXCEPT ![candidate] = 2]
    /\ UNCHANGED <<active, run, profiles, evidence>>

StartBenchmark ==
    /\ active < 2 /\ run = NoRun
    /\ run' = [candidate |-> active + 1, epoch |-> versions[active + 1]]
    /\ UNCHANGED <<active, versions, profiles, evidence>>

FinishBenchmark(sufficient) ==
    /\ run # NoRun
    /\ profiles' = [profiles EXCEPT ![run.candidate] =
                    [epoch |-> run.epoch, fresh |-> TRUE, covers |-> sufficient]]
    /\ run' = NoRun
    /\ UNCHANGED <<active, versions, evidence>>

Expire(candidate) ==
    /\ profiles[candidate].fresh
    /\ profiles' = [profiles EXCEPT ![candidate].fresh = FALSE]
    /\ UNCHANGED <<active, versions, run, evidence>>

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
    /\ UNCHANGED <<versions, run, profiles>>

Next ==
    \/ StartBenchmark \/ Switch
    \/ \E candidate \in Candidates : ChangeEpoch(candidate) \/ Expire(candidate)
    \/ \E sufficient \in BOOLEAN : FinishBenchmark(sufficient)

Spec == Init /\ [][Next]_vars
RequiresMeasurement == evidence.measured
NeedsCurrentEpoch == evidence.exact
NeedsFreshProfile == evidence.fresh
CoversArrivalRate == evidence.covers
====
