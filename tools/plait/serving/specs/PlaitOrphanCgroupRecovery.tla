---- MODULE PlaitOrphanCgroupRecovery ----
EXTENDS Naturals

CONSTANTS RequireExactBinding, RequireRevocation, RequireEmpty, ScopeKill

VARIABLES owner, exact, revoked, oldTasks, siblingTasks, killed,
          safeKill, journal, safeJournal, released
vars == <<owner, exact, revoked, oldTasks, siblingTasks, killed,
          safeKill, journal, safeJournal, released>>

Init ==
    /\ owner = TRUE /\ exact = TRUE /\ revoked = FALSE
    /\ oldTasks = 2 /\ siblingTasks = 1 /\ killed = FALSE
    /\ safeKill = TRUE /\ journal = FALSE /\ safeJournal = TRUE /\ released = FALSE

LoseOwner ==
    /\ owner /\ owner' = FALSE
    /\ UNCHANGED <<exact, revoked, oldTasks, siblingTasks, killed,
                   safeKill, journal, safeJournal, released>>

ChangeBinding ==
    /\ ~owner /\ exact /\ exact' = FALSE
    /\ UNCHANGED <<owner, revoked, oldTasks, siblingTasks, killed,
                   safeKill, journal, safeJournal, released>>

Revoke ==
    /\ ~revoked /\ revoked' = TRUE
    /\ UNCHANGED <<owner, exact, oldTasks, siblingTasks, killed,
                   safeKill, journal, safeJournal, released>>

Kill ==
    /\ ~owner /\ oldTasks > 0 /\ ~killed
    /\ (exact \/ ~RequireExactBinding)
    /\ (revoked \/ ~RequireRevocation)
    /\ killed' = TRUE
    /\ safeKill' = (exact /\ revoked)
    /\ siblingTasks' = IF ScopeKill THEN siblingTasks ELSE 0
    /\ UNCHANGED <<owner, exact, revoked, oldTasks, journal, safeJournal, released>>

TaskExits ==
    /\ oldTasks > 0 /\ oldTasks' = oldTasks - 1
    /\ UNCHANGED <<owner, exact, revoked, siblingTasks, killed,
                   safeKill, journal, safeJournal, released>>

JournalExit ==
    /\ ~owner /\ revoked /\ ~journal
    /\ (exact \/ ~RequireExactBinding)
    /\ (oldTasks = 0 \/ ~RequireEmpty)
    /\ journal' = TRUE
    /\ safeJournal' = (exact /\ oldTasks = 0)
    /\ UNCHANGED <<owner, exact, revoked, oldTasks, siblingTasks, killed,
                   safeKill, released>>

Release ==
    /\ journal /\ ~released /\ released' = TRUE
    /\ UNCHANGED <<owner, exact, revoked, oldTasks, siblingTasks, killed,
                   safeKill, safeJournal, journal>>

Next == LoseOwner \/ ChangeBinding \/ Revoke \/ Kill \/ TaskExits \/ JournalExit \/ Release
Spec == Init /\ [][Next]_vars
KillUsesExactRevokedTree == safeKill
IndependentTreeSurvives == siblingTasks = 1
ReleaseHasEmptyTreeProof == released => safeJournal /\ oldTasks = 0
====
