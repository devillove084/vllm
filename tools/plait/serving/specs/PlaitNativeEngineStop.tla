---- MODULE PlaitNativeEngineStop ----
EXTENDS Naturals, FiniteSets

CONSTANTS RequireGroupExit, RequireGrace, ScopeSignals, PreserveReleaseId

Groups == {"old", "independent"}
Grants == 1..2
Send(record, id) == [grant |-> record, operationId |-> id]

VARIABLES revoked, stopStep, elapsed, leaderAlive, childAlive, independentAlive,
          journal, launcher, sent, applied, acknowledged, finished
vars == <<revoked, stopStep, elapsed, leaderAlive, childAlive, independentAlive,
          journal, launcher, sent, applied, acknowledged, finished>>

Init ==
    /\ revoked = FALSE /\ stopStep = 0 /\ elapsed = FALSE
    /\ leaderAlive = TRUE /\ childAlive = TRUE /\ independentAlive = TRUE
    /\ journal = FALSE /\ launcher = TRUE
    /\ sent = {} /\ applied = {} /\ acknowledged = {}
    /\ finished = FALSE

Revoke ==
    /\ ~revoked
    /\ revoked' = TRUE
    /\ UNCHANGED <<stopStep, elapsed, leaderAlive, childAlive, independentAlive,
                   journal, launcher, sent, applied, acknowledged, finished>>

Signal(group, step) ==
    /\ launcher /\ revoked /\ (leaderAlive \/ childAlive)
    /\ group = "old" \/ ~ScopeSignals
    /\ stopStep = step - 1
    /\ step = 1 \/ elapsed \/ ~RequireGrace
    /\ stopStep' = step
    /\ independentAlive' = (independentAlive /\ group # "independent")
    /\ UNCHANGED <<revoked, elapsed, leaderAlive, childAlive, journal,
                   launcher, sent, applied, acknowledged, finished>>

GraceExpires ==
    /\ stopStep >= 1 /\ ~elapsed
    /\ elapsed' = TRUE
    /\ UNCHANGED <<revoked, stopStep, leaderAlive, childAlive, independentAlive,
                   journal, launcher, sent, applied, acknowledged, finished>>

LeaderExits ==
    /\ stopStep >= 1 /\ leaderAlive
    /\ leaderAlive' = FALSE
    /\ UNCHANGED <<revoked, stopStep, elapsed, childAlive, independentAlive,
                   journal, launcher, sent, applied, acknowledged, finished>>

StalledChildExits ==
    /\ stopStep = 2 /\ childAlive
    /\ childAlive' = FALSE
    /\ UNCHANGED <<revoked, stopStep, elapsed, leaderAlive, independentAlive,
                   journal, launcher, sent, applied, acknowledged, finished>>

RecordExit ==
    /\ launcher /\ revoked /\ ~journal /\ ~finished /\ ~leaderAlive
    /\ ~childAlive \/ ~RequireGroupExit
    /\ journal' = TRUE
    /\ UNCHANGED <<revoked, stopStep, elapsed, leaderAlive, childAlive,
                   independentAlive, launcher, sent, applied, acknowledged, finished>>

Release(grant) ==
    /\ launcher /\ journal /\ grant \notin acknowledged
    /\ LET id == IF PreserveReleaseId \/ Send(grant, grant) \notin sent
                THEN grant ELSE grant + Cardinality(Grants)
       IN sent' = sent \cup {Send(grant, id)}
    /\ applied' = applied \cup {grant}
    /\ UNCHANGED <<revoked, stopStep, elapsed, leaderAlive, childAlive,
                   independentAlive, journal, launcher, acknowledged, finished>>

Receipt(grant) ==
    /\ launcher /\ grant \in applied /\ grant \notin acknowledged
    /\ acknowledged' = acknowledged \cup {grant}
    /\ UNCHANGED <<revoked, stopStep, elapsed, leaderAlive, childAlive,
                   independentAlive, journal, launcher, sent, applied, finished>>

Crash ==
    /\ launcher /\ ~finished
    /\ launcher' = FALSE /\ acknowledged' = {}
    /\ UNCHANGED <<revoked, stopStep, elapsed, leaderAlive, childAlive,
                   independentAlive, journal, sent, applied, finished>>

Resume ==
    /\ ~launcher /\ journal /\ ~finished
    /\ launcher' = TRUE
    /\ UNCHANGED <<revoked, stopStep, elapsed, leaderAlive, childAlive,
                   independentAlive, journal, sent, applied, acknowledged, finished>>

RemoveRecord ==
    /\ launcher /\ journal /\ Grants \subseteq acknowledged
    /\ journal' = FALSE /\ finished' = TRUE
    /\ UNCHANGED <<revoked, stopStep, elapsed, leaderAlive, childAlive,
                   independentAlive, launcher, sent, applied, acknowledged>>

Next ==
    \/ Revoke \/ GraceExpires \/ LeaderExits \/ StalledChildExits \/ RecordExit
    \/ Crash \/ Resume \/ RemoveRecord
    \/ \E group \in Groups, step \in 1..2 : Signal(group, step)
    \/ \E grant \in Grants : Release(grant) \/ Receipt(grant)

Spec == Init /\ [][Next]_vars
SignalsStayInGroup == independentAlive
ForceWaitsForGrace == stopStep = 2 => elapsed
ReleaseRequiresProcessGroupExit ==
    applied # {} => revoked /\ ~leaderAlive /\ ~childAlive /\ (journal \/ finished)
ReplayKeepsReleaseId == \A message \in sent : message.operationId = message.grant
RemoveAfterReceipts == finished => Grants \subseteq acknowledged
====
