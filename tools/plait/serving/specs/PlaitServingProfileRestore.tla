---- MODULE PlaitServingProfileRestore ----
\* A pair of writers may finish valid benchmarks for the same identity.
\* The lock covers the read, merge and publish. Publish is the durable
\* boundary of atomic replacement and sync. Incomplete staging is private.
\* The reader's cache is volatile. A restore and each later use revalidate
\* the evidence. One field can drift in a run, and the reader restarts once.
\* Timing, fingerprints and HTTP results are abstracted.
EXTENDS Naturals

Prefill == 1
Decode == 2
Roles == {Prefill, Decode}
Writers == {1, 2}
Changes == {"Model", "Route", "Configuration", "Epoch", "Origin", "Manifest"}
Context(model, route, settings) == [model |-> model, route |-> route, settings |-> settings]
Identity(change) ==
    LET epoch == IF change = "Epoch" THEN 2 ELSE 1
    IN [context |-> Context(IF change = "Model" THEN 2 ELSE 1,
                             IF change = "Route" THEN 2 ELSE 1,
                             IF change = "Configuration" THEN 2 ELSE 1),
        versions |-> {<<role, epoch>> : role \in Roles},
        binding |-> <<10, IF change = "Origin" THEN 21 ELSE 20>>,
        manifest |-> IF change = "Manifest" THEN 2 ELSE 1]

Profile(id, expiry) == [identity |-> id, expiry |-> expiry]
NoProfile == Profile(Identity("None"), 0)
Writer(phase, entry) == [phase |-> phase, entry |-> entry]
PrivateWrite(holder, image, askedAt) == [holder |-> holder, image |-> image, askedAt |-> askedAt]
Unlocked == PrivateWrite(0, {}, 0)
Reader(boot, loaded, pending, cached, used) ==
    [boot |-> boot, loaded |-> loaded, pending |-> pending, cached |-> cached, used |-> used]

VARIABLES clock, view, writers, staged, disk, reader
vars == <<clock, view, writers, staged, disk, reader>>

Init ==
    /\ clock = 0
    /\ view = [change |-> "None", active |-> {}, health |-> Roles,
                phase |-> [r \in Roles |-> "Ready"]]
    /\ writers = [w \in Writers |-> Writer("Idle", NoProfile)]
    /\ staged = Unlocked /\ disk = {}
    /\ reader = Reader(0, FALSE, {}, NoProfile, "None")

Current == Identity(view.change)
PhaseMatches == \A r \in Roles : view.phase[r] = (IF r \in view.active THEN "Active" ELSE "Ready")
Healthy == view.health = Roles
BeforeExpiry(entry) == entry.expiry > clock

\* This transition represents acceptance of a complete, positive calibration.
\* Both candidates have an exact identity and are healthy, ready and private.
AcceptCalibration(w) ==
    /\ writers[w].phase = "Idle" /\ view.active = {} /\ PhaseMatches /\ Healthy
    /\ w + 1 > clock
    /\ writers' = [writers EXCEPT ![w] = Writer("Validated", Profile(Current, w + 1))]
    /\ UNCHANGED <<clock, view, staged, disk, reader>>

Merged(incoming) ==
    IF \E r \in disk : BeforeExpiry(r) /\ r.identity = incoming.identity /\ r.expiry > incoming.expiry
    THEN disk
    ELSE {incoming} \cup {r \in disk : BeforeExpiry(r) /\ r.identity # incoming.identity}

StageWrite(w) ==
    /\ staged = Unlocked /\ writers[w].phase = "Validated"
    /\ staged' = PrivateWrite(w, Merged(writers[w].entry), clock)
    /\ UNCHANGED <<clock, view, writers, disk, reader>>

PublishWrite ==
    /\ staged.holder \in Writers
    /\ disk' = staged.image
    /\ writers' = [writers EXCEPT ![staged.holder].phase = "Published"]
    /\ staged' = Unlocked
    /\ UNCHANGED <<clock, view, reader>>

WriterCrash ==
    /\ staged.holder \in Writers
    /\ writers' = [writers EXCEPT ![staged.holder].phase = "Crashed"]
    /\ staged' = Unlocked
    /\ UNCHANGED <<clock, view, disk, reader>>

ReadFile ==
    /\ ~reader.loaded
    /\ reader' = [reader EXCEPT !.loaded = TRUE, !.pending = {r \in disk : r.identity.context = Current.context}]
    /\ UNCHANGED <<clock, view, writers, staged, disk>>

Compatible(entry) ==
    /\ entry # NoProfile /\ BeforeExpiry(entry) /\ entry.identity = Current
    /\ PhaseMatches /\ Healthy

Restore(entry) ==
    /\ entry \in reader.pending /\ Compatible(entry)
    /\ reader' = [reader EXCEPT !.pending = @ \ {entry}, !.cached = entry]
    /\ UNCHANGED <<clock, view, writers, staged, disk>>

Restart ==
    /\ reader.boot = 0
    /\ reader' = Reader(1, FALSE, {}, NoProfile, "None")
    /\ UNCHANGED <<clock, view, writers, staged, disk>>

UseProfile ==
    /\ Compatible(reader.cached)
    /\ reader' = [reader EXCEPT !.used = IF view.active = Roles THEN "Active" ELSE "Candidate"]
    /\ UNCHANGED <<clock, view, writers, staged, disk>>

Cutover ==
    /\ view.active = {} /\ Compatible(reader.cached)
    /\ view' = [view EXCEPT !.active = Roles, !.phase = [r \in Roles |-> "Active"]]
    /\ UNCHANGED <<clock, writers, staged, disk, reader>>

IdentityChange(change) ==
    /\ view.change = "None"
    /\ view' = [view EXCEPT !.change = change]
    /\ UNCHANGED <<clock, writers, staged, disk, reader>>

HttpDown(r) ==
    /\ r \in view.health
    /\ view' = [view EXCEPT !.health = @ \ {r}]
    /\ UNCHANGED <<clock, writers, staged, disk, reader>>

HttpUp(r) ==
    /\ r \notin view.health /\ view.phase[r] \in {"Ready", "Active"}
    /\ view' = [view EXCEPT !.health = @ \cup {r}]
    /\ UNCHANGED <<clock, writers, staged, disk, reader>>

Failure(r) ==
    /\ view.phase[r] \in {"Ready", "Active"}
    /\ view' = [view EXCEPT !.phase[r] = "Revoked", !.active = @ \ {r}]
    /\ UNCHANGED <<clock, writers, staged, disk, reader>>

AdvanceClock ==
    /\ clock < 3
    /\ clock' = clock + 1
    /\ UNCHANGED <<view, writers, staged, disk, reader>>

Next ==
    \/ \E w \in Writers : AcceptCalibration(w) \/ StageWrite(w)
    \/ PublishWrite \/ WriterCrash \/ ReadFile \/ Restart \/ UseProfile \/ Cutover \/ AdvanceClock
    \/ \E entry \in reader.pending : Restore(entry)
    \/ \E change \in Changes : IdentityChange(change)
    \/ \E r \in Roles : HttpDown(r) \/ HttpUp(r) \/ Failure(r)

Spec == Init /\ [][Next]_vars

DurableWritesAreAtomic ==
    /\ \A w \in Writers : ~ENABLED (StageWrite(w) /\ disk' # disk)
    /\ ~ENABLED ((WriterCrash \/ Restart) /\ disk' # disk)
    /\ ~ENABLED (Restart /\ (reader'.cached # NoProfile \/ reader'.pending # {} \/ reader'.loaded))
    /\ ~ENABLED (PublishWrite /\ disk' # staged.image)

OnlyCalibratedRecords ==
    /\ \A p \in disk : \E w \in Writers : writers[w].entry = p /\ writers[w].phase = "Published"
    /\ \A p, q \in disk : p.identity = q.identity => p = q
    /\ \A w \in Writers :
         ~ENABLED (AcceptCalibration(w) /\ (view.active # {} \/ ~PhaseMatches \/ ~Healthy \/
             writers'[w].entry.identity # Current \/ writers'[w].entry.expiry <= clock))

EarlierWriterCannotShortenExpiry ==
    staged.holder \in Writers =>
        LET candidate == writers[staged.holder].entry IN
        \A r \in disk :
            r.identity = candidate.identity /\ r.expiry > staged.askedAt /\ r.expiry > candidate.expiry =>
                ~ENABLED (PublishWrite /\ r \notin disk')

\* Spell out the independently checked fields so omitting a comparison in
\* Compatible cannot also relax the restore or use contract.
CurrentEvidence(entry) ==
    /\ entry.expiry > clock
    /\ entry.identity.context = Current.context
    /\ entry.identity.versions = Current.versions
    /\ entry.identity.binding = Current.binding
    /\ entry.identity.manifest = Current.manifest
    /\ \A r \in Roles : r \in view.health /\
           IF r \in view.active THEN view.phase[r] = "Active" ELSE view.phase[r] = "Ready"

RestoreChecksCurrentEvidence ==
    \A entry \in reader.pending :
        /\ ~ENABLED (Restore(entry) /\ (~CurrentEvidence(entry) \/ reader'.cached # entry))
        /\ CurrentEvidence(entry) => ENABLED (Restore(entry) /\ reader'.cached = entry)

UseRechecksCurrentEvidence ==
    ~ENABLED ((UseProfile \/ Cutover) /\ ~CurrentEvidence(reader.cached))

MeasuredActiveBaselineIsAvailable ==
    reader.cached # NoProfile /\ view.active = Roles /\ CurrentEvidence(reader.cached) =>
        ENABLED (UseProfile /\ reader'.used = "Active")
====
