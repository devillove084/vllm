---- MODULE PlaitFailedLaunchRecovery ----
EXTENDS Naturals, FiniteSets

CONSTANTS PersistBeforeRpc, PreserveOperationIds, FenceReleases

Report == 1
Devices == 1..2
Release(device) == device + 1
Releases == {Release(device) : device \in Devices}
Ops == {Report} \cup Releases
Attempt(op, id) == [operation |-> op, id |-> id]

VARIABLES journal, launcher, locked, attempted, applied, acknowledged,
          successorGrants, lostSuccessor, done
vars == <<journal, launcher, locked, attempted, applied, acknowledged,
          successorGrants, lostSuccessor, done>>

Init ==
    \* The first exec was definitively rejected while holding the local lock.
    /\ journal = FALSE
    /\ launcher = TRUE /\ locked = TRUE
    /\ attempted = {} /\ applied = {} /\ acknowledged = {}
    /\ successorGrants = {} /\ lostSuccessor = FALSE
    /\ done = FALSE

Persist ==
    /\ launcher /\ locked /\ ~journal /\ ~done
    /\ journal' = TRUE
    /\ UNCHANGED <<launcher, locked, attempted, applied, acknowledged,
                   successorGrants, lostSuccessor, done>>

Send(op) ==
    /\ launcher /\ locked /\ ~done
    /\ journal \/ ~PersistBeforeRpc
    /\ IF op = Report THEN Report \notin applied ELSE Report \in applied
    /\ LET id == IF PreserveOperationIds \/ Attempt(op, op) \notin attempted
                THEN op ELSE op + Cardinality(Ops)
       IN attempted' = attempted \cup {Attempt(op, id)}
    /\ applied' = applied \cup {op}
    /\ lostSuccessor' = (lostSuccessor \/
                         (~FenceReleases /\ op \in Releases /\ op - 1 \in successorGrants))
    /\ UNCHANGED <<journal, launcher, locked, acknowledged, successorGrants, done>>

Ack(op) ==
    /\ launcher /\ op \in applied /\ op \notin acknowledged
    /\ acknowledged' = acknowledged \cup {op}
    /\ UNCHANGED <<journal, launcher, locked, attempted, applied,
                   successorGrants, lostSuccessor, done>>

Crash ==
    /\ launcher /\ ~done
    /\ launcher' = FALSE /\ locked' = FALSE /\ acknowledged' = {}
    /\ UNCHANGED <<journal, attempted, applied, successorGrants, lostSuccessor, done>>

Recover ==
    /\ ~launcher /\ journal /\ ~done
    /\ launcher' = TRUE /\ locked' = TRUE
    /\ UNCHANGED <<journal, attempted, applied, acknowledged,
                   successorGrants, lostSuccessor, done>>

Reuse(device) ==
    /\ Release(device) \in applied
    /\ device \notin successorGrants
    /\ successorGrants' = successorGrants \cup {device}
    /\ UNCHANGED <<journal, launcher, locked, attempted, applied,
                   acknowledged, lostSuccessor, done>>

Finish ==
    /\ launcher /\ Releases \subseteq acknowledged /\ ~done
    /\ journal' = FALSE /\ locked' = FALSE /\ launcher' = FALSE /\ done' = TRUE
    /\ UNCHANGED <<attempted, applied, acknowledged, successorGrants, lostSuccessor>>

Next ==
    \/ Persist \/ Crash \/ Recover \/ Finish
    \/ \E op \in Ops : Send(op) \/ Ack(op)
    \/ \E device \in Devices : Reuse(device)

Spec == Init /\ [][Next]_vars
DurableBeforeMutation == attempted # {} => journal \/ done
RetryUsesOriginalId == \A attempt \in attempted : attempt.id = attempt.operation
SuccessorGrantsAreFenced == ~lostSuccessor
ReleasesFollowRevocation == \A op \in Releases : op \in applied => Report \in applied
RecoveryHoldsLock == launcher => locked
DeleteAfterEveryReceipt == done => Releases \subseteq acknowledged
====
