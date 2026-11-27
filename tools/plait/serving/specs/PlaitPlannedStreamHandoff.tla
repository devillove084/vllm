---- MODULE PlaitPlannedStreamHandoff ----
EXTENDS Naturals

\* The source can compute ahead. It remains live while the replacement warms.
\* The ingress transfers the last visible position and grants one connection
\* authority to publish the suffix. A seeded source has already admitted and
\* acknowledged the seed. A seeded target must support the same sampler
\* and acknowledge the same seed before it can emit any suffix.
CONSTANT SeededRequest
Limit == 4
VARIABLES routed, ready, resume, sameSampler, seedConfirmed,
          produced, pages, visible, checkpoint, imported, phase
vars == <<routed, ready, resume, sameSampler, seedConfirmed,
          produced, pages, visible, checkpoint, imported, phase>>

CanCut == ~SeededRequest \/ (resume /\ sameSampler)

Init ==
    /\ routed = TRUE
    /\ ready = FALSE
    /\ resume \in BOOLEAN
    /\ sameSampler \in (IF SeededRequest /\ resume THEN BOOLEAN ELSE {FALSE})
    /\ seedConfirmed = FALSE
    /\ produced = 0
    /\ pages = 0
    /\ visible = 0
    /\ checkpoint = 0
    /\ imported = 0
    /\ phase = "source"

Warm ==
    /\ ~ready
    /\ ready' = TRUE
    /\ UNCHANGED <<routed, resume, sameSampler, seedConfirmed,
                   produced, pages, visible, checkpoint, imported, phase>>

RemoveSource ==
    /\ routed
    /\ routed' = FALSE
    /\ UNCHANGED <<ready, resume, sameSampler, seedConfirmed,
                   produced, pages, visible, checkpoint, imported, phase>>

Compute ==
    /\ phase \in {"source", "cut", "target"}
    /\ produced < Limit
    /\ produced' = produced + 1
    /\ UNCHANGED <<routed, ready, resume, sameSampler, seedConfirmed,
                   pages, visible, checkpoint, imported, phase>>

Store ==
    /\ pages < produced
    /\ pages' = pages + 1
    /\ UNCHANGED <<routed, ready, resume, sameSampler, seedConfirmed,
                   produced, visible, checkpoint, imported, phase>>

EmitSource ==
    /\ phase = "source"
    /\ visible < produced
    /\ visible' = visible + 1
    /\ UNCHANGED <<routed, ready, resume, sameSampler, seedConfirmed,
                   produced, pages, checkpoint, imported, phase>>

Cut ==
    /\ phase = "source"
    /\ 0 < visible /\ visible < Limit
    /\ ~routed
    /\ ready
    /\ CanCut
    /\ checkpoint' = visible
    /\ phase' = "cut"
    /\ UNCHANGED <<routed, ready, resume, sameSampler, seedConfirmed,
                   produced, pages, visible, imported>>

Verify(ack, matches, hit, seedMatch) ==
    /\ phase = "cut"
    /\ hit <= pages /\ hit <= checkpoint
    /\ imported' = IF resume THEN hit ELSE 0
    /\ seedConfirmed' = seedMatch
    /\ phase' = IF matches /\ (~resume \/ checkpoint = ack)
                   /\ (~SeededRequest \/ seedMatch)
                THEN "target" ELSE "failed"
    /\ UNCHANGED <<routed, ready, resume, sameSampler, produced, pages, visible, checkpoint>>

EmitTarget ==
    /\ phase = "target"
    /\ visible >= checkpoint /\ visible < Limit
    /\ visible' = visible + 1
    /\ UNCHANGED <<routed, ready, resume, sameSampler, seedConfirmed,
                   produced, pages, checkpoint, imported, phase>>

Finish ==
    /\ phase \in {"source", "target"}
    /\ visible = Limit
    /\ phase' = "done"
    /\ UNCHANGED <<routed, ready, resume, sameSampler, seedConfirmed,
                   produced, pages, visible, checkpoint, imported>>

Next == Warm \/ RemoveSource \/ Compute \/ Store \/ EmitSource \/ Cut
    \/ (\E ack \in 0..Limit, matches \in BOOLEAN, hit \in 0..Limit,
           seedMatch \in (IF SeededRequest THEN BOOLEAN ELSE {FALSE}) :
            Verify(ack, matches, hit, seedMatch))
    \/ EmitTarget \/ Finish
Spec == Init /\ [][Next]_vars

PreservesCut == checkpoint <= visible
ImportWithinVisible == imported <= checkpoint /\ imported <= pages
TerminalOnceComplete == phase = "done" => visible = Limit
SeedConfirmedBeforeTarget == SeededRequest /\ checkpoint > 0 /\ phase \in {"target", "done"}
    => seedConfirmed
CutAtVisible == [][(phase = "source" /\ phase' = "cut") =>
    (ready /\ ~routed /\ CanCut /\ checkpoint' = visible)]_vars
OneOutputOwner == [][(visible' > visible) =>
    (visible' = visible + 1 /\ phase \in {"source", "target"})]_vars
====
