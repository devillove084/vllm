---- MODULE PlaitNativeLaunchFailure ----
EXTENDS Naturals

CONSTANTS CheckAfterLock, WaitForExit, RetainFailedLock, FenceFault

Slots == 1..2
Phases == {"Preparing", "Revoked", "Retired"}
LocalStages == {"Idle", "Locked", "Checked", "Running", "Exited",
                "ExecRejected", "Released"}

VARIABLES phase, local, locks, released, staleCheck, incumbentAlive
vars == <<phase, local, locks, released, staleCheck, incumbentAlive>>

Init ==
    /\ phase = "Preparing"
    /\ local = [slot \in Slots |-> "Idle"]
    /\ locks = {}
    /\ released = {}
    /\ staleCheck = FALSE
    /\ incumbentAlive = TRUE

Lock(slot) ==
    \* A stale placement read may still say Preparing after remote revocation.
    /\ local[slot] = "Idle" /\ slot \notin locks
    /\ local' = [local EXCEPT ![slot] = "Locked"]
    /\ locks' = locks \cup {slot}
    /\ UNCHANGED <<phase, released, staleCheck, incumbentAlive>>

Check(slot) ==
    /\ local[slot] = "Locked"
    /\ IF phase = "Preparing" \/ ~CheckAfterLock
       THEN /\ local' = [local EXCEPT ![slot] = "Checked"]
            /\ staleCheck' = (staleCheck \/ phase # "Preparing")
            /\ UNCHANGED locks
       ELSE /\ local' = [local EXCEPT ![slot] = "Idle"]
            /\ locks' = locks \ {slot}
            /\ UNCHANGED staleCheck
    /\ UNCHANGED <<phase, released, incumbentAlive>>

Exec(slot) ==
    \* A different slot can report failure immediately after this slot's check.
    /\ local[slot] = "Checked"
    /\ local' = [local EXCEPT ![slot] = "Running"]
    /\ UNCHANGED <<phase, locks, released, staleCheck, incumbentAlive>>

RejectExec(slot) ==
    /\ local[slot] = "Checked"
    /\ local' = [local EXCEPT ![slot] = "ExecRejected"]
    /\ locks' = IF RetainFailedLock THEN locks ELSE locks \ {slot}
    /\ UNCHANGED <<phase, released, staleCheck, incumbentAlive>>

Exit(slot) ==
    /\ local[slot] = "Running"
    /\ local' = [local EXCEPT ![slot] = "Exited"]
    /\ locks' = locks \ {slot}
    /\ UNCHANGED <<phase, released, staleCheck, incumbentAlive>>

Report(slot) ==
    /\ phase = "Preparing"
    /\ local[slot] \in {"Exited", "ExecRejected"}
    /\ phase' = "Revoked"
    /\ incumbentAlive' = IF FenceFault THEN incumbentAlive ELSE FALSE
    /\ UNCHANGED <<local, locks, released, staleCheck>>

Release(slot) ==
    /\ phase = "Revoked" /\ slot \notin released
    /\ (local[slot] \in {"Exited", "ExecRejected"}
        \/ (~WaitForExit /\ local[slot] = "Running"))
    /\ released' = released \cup {slot}
    /\ phase' = IF released' = Slots THEN "Retired" ELSE phase
    /\ UNCHANGED <<local, locks, staleCheck, incumbentAlive>>

AckRelease(slot) ==
    \* The release can commit while the launcher is still retrying its reply.
    /\ slot \in released
    /\ local[slot] \in {"Exited", "ExecRejected"}
    /\ local' = [local EXCEPT ![slot] = "Released"]
    /\ locks' = locks \ {slot}
    /\ UNCHANGED <<phase, released, staleCheck, incumbentAlive>>

Next ==
    \E slot \in Slots :
        Lock(slot) \/ Check(slot) \/ Exec(slot) \/ RejectExec(slot)
        \/ Exit(slot) \/ Report(slot) \/ Release(slot) \/ AckRelease(slot)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ local \in [Slots -> LocalStages]
    /\ locks \subseteq Slots /\ released \subseteq Slots
    /\ staleCheck \in BOOLEAN /\ incumbentAlive \in BOOLEAN

FreshLaunchCheck == ~staleCheck
NoReleasedEngine == \A slot \in released : local[slot] # "Running"
FailedLaunchRetainsClaim ==
    \A slot \in Slots : local[slot] = "ExecRejected" => slot \in locks
IncumbentStaysActive == incumbentAlive
RetiredAfterBothReleases == phase = "Retired" => released = Slots
====
