---- MODULE PlaitEngineOwnerRecovery ----
EXTENDS Naturals

CONSTANTS RetainOwnerLock, RecordBeforeExec, MatchStopOrigin, RequireExitRecord

VARIABLES supervisor, ownerLock, owners, recordedOwner, started, running,
          revoked, request, stoppedBy, exitRecord, released
vars == <<supervisor, ownerLock, owners, recordedOwner, started, running,
          revoked, request, stoppedBy, exitRecord, released>>

Init ==
    /\ supervisor = TRUE /\ ownerLock = FALSE /\ owners = 0 /\ recordedOwner = FALSE
    /\ started = FALSE /\ running = FALSE /\ revoked = FALSE
    /\ request = "none" /\ stoppedBy = "none" /\ exitRecord = FALSE /\ released = FALSE

StartOwner ==
    /\ supervisor /\ ~ownerLock /\ ~recordedOwner /\ ~revoked /\ owners < 2
    /\ ownerLock' = TRUE /\ owners' = owners + 1
    /\ UNCHANGED <<supervisor, recordedOwner, started, running, revoked,
                   request, stoppedBy, exitRecord, released>>

RecordOwner ==
    /\ owners = 1 /\ ownerLock /\ ~recordedOwner
    /\ recordedOwner' = TRUE
    /\ UNCHANGED <<supervisor, ownerLock, owners, started, running,
                   revoked, request, stoppedBy, exitRecord, released>>

Exec ==
    /\ owners = 1 /\ ~started /\ (recordedOwner \/ ~RecordBeforeExec)
    /\ started' = TRUE /\ running' = TRUE
    /\ UNCHANGED <<supervisor, ownerLock, owners, recordedOwner, revoked,
                   request, stoppedBy, exitRecord, released>>

LoseSupervisor ==
    /\ supervisor /\ ~released
    /\ supervisor' = FALSE
    /\ ownerLock' = (ownerLock /\ RetainOwnerLock)
    /\ UNCHANGED <<owners, recordedOwner, started, running, revoked,
                   request, stoppedBy, exitRecord, released>>

ResumeSupervisor ==
    /\ ~supervisor
    /\ supervisor' = TRUE
    /\ UNCHANGED <<ownerLock, owners, recordedOwner, started, running, revoked,
                   request, stoppedBy, exitRecord, released>>

Revoke ==
    /\ ~revoked
    /\ revoked' = TRUE
    /\ UNCHANGED <<supervisor, ownerLock, owners, recordedOwner, started, running,
                   request, stoppedBy, exitRecord, released>>

WriteStop(origin) ==
    /\ request # origin
    /\ IF origin = "stale" THEN request = "none"
       ELSE supervisor /\ revoked /\ recordedOwner
    /\ request' = origin
    /\ UNCHANGED <<supervisor, ownerLock, owners, recordedOwner, started, running,
                   revoked, stoppedBy, exitRecord, released>>

EngineExits ==
    /\ running /\ request # "none"
    /\ (request = "current" \/ ~MatchStopOrigin)
    /\ running' = FALSE /\ stoppedBy' = request
    /\ UNCHANGED <<supervisor, ownerLock, owners, recordedOwner, started, revoked,
                   request, exitRecord, released>>

RecordExit ==
    /\ owners = 1 /\ started /\ ~running /\ ~exitRecord
    /\ exitRecord' = TRUE /\ ownerLock' = FALSE /\ owners' = 0
    /\ UNCHANGED <<supervisor, recordedOwner, started, running, revoked,
                   request, stoppedBy, released>>

Release ==
    /\ supervisor /\ revoked /\ recordedOwner /\ ~released
    /\ (exitRecord \/ ~RequireExitRecord)
    /\ released' = TRUE
    /\ UNCHANGED <<supervisor, ownerLock, owners, recordedOwner, started, running, revoked,
                   request, stoppedBy, exitRecord>>

Next == StartOwner \/ RecordOwner \/ Exec \/ LoseSupervisor \/ ResumeSupervisor
        \/ Revoke \/ EngineExits \/ RecordExit \/ Release
        \/ \E origin \in {"current", "stale"} : WriteStop(origin)

Spec == Init /\ [][Next]_vars
SingleOwner == owners <= 1
ExecHasDurableOwner == started => recordedOwner
StopBindsInstance == stoppedBy # "none" => stoppedBy = "current" /\ revoked
ReleaseRequiresExit == released => exitRecord /\ ~running /\ owners = 0
====
