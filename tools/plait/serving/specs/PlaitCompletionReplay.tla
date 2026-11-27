---- MODULE PlaitCompletionReplay ----
EXTENDS Naturals, Sequences

CONSTANT UseCheckpoint

\* A value represents an output token and its corresponding text.
Original == << <<1, "a">>, <<2, "b">>, <<3, "c">> >>
Candidates == {Original,
    << <<1, "a">>, <<2, "b">>, <<4, "d">> >>,
    << <<1, "x">>, <<2, "b">>, <<3, "c">> >>,
    << <<9, "a">>, <<2, "b">>, <<3, "c">> >>,
    << <<1, "a">> >>}

VARIABLES active, attempt, output, cursor, visible, phase, checkpoint, acknowledged
vars == <<active, attempt, output, cursor, visible, phase, checkpoint, acknowledged>>
Acknowledgments == IF UseCheckpoint THEN 0..(Len(Original) + 1) ELSE {0}

Init ==
    /\ active = 1 /\ attempt = 1 /\ output = Original
    /\ cursor = 0 /\ visible = <<>> /\ phase = "Reading"
    /\ checkpoint = 0 /\ acknowledged = 0

Activate ==
    /\ active < 3 /\ active' = active + 1
    /\ UNCHANGED <<attempt, output, cursor, visible, phase, checkpoint, acknowledged>>

Interrupt ==
    /\ phase = "Reading" /\ phase' = "Waiting"
    /\ UNCHANGED <<active, attempt, output, cursor, visible, checkpoint, acknowledged>>

Replay(candidate, count) ==
    /\ phase = "Waiting" /\ active > attempt
    /\ attempt' = active /\ output' = candidate
    /\ cursor' = 0
    /\ checkpoint' = IF UseCheckpoint THEN Len(visible) ELSE 0
    /\ acknowledged' = count
    /\ phase' = IF checkpoint' = 0 \/ count = checkpoint' THEN "Reading" ELSE "Failed"
    /\ UNCHANGED <<active, visible>>

Match ==
    /\ phase = "Reading"
    /\ cursor < Len(visible) /\ cursor < Len(output)
    /\ output[cursor + 1] = visible[cursor + 1].value
    /\ cursor' = cursor + 1
    /\ UNCHANGED <<active, attempt, output, visible, phase, checkpoint, acknowledged>>

Publish ==
    /\ phase = "Reading"
    /\ cursor = Len(visible) /\ cursor < Len(output)
    /\ visible' = Append(visible, [position |-> cursor + 1, value |-> output[cursor + 1]])
    /\ cursor' = cursor + 1
    /\ UNCHANGED <<active, attempt, output, phase, checkpoint, acknowledged>>

Reject ==
    /\ phase = "Reading" /\ cursor < Len(visible)
    /\ (IF cursor = Len(output) THEN TRUE
        ELSE output[cursor + 1] # visible[cursor + 1].value)
    /\ phase' = "Failed"
    /\ UNCHANGED <<active, attempt, output, cursor, visible, checkpoint, acknowledged>>

Finish ==
    /\ phase = "Reading" /\ cursor = Len(output) /\ cursor = Len(visible)
    /\ phase' = "Done"
    /\ UNCHANGED <<active, attempt, output, cursor, visible, checkpoint, acknowledged>>

Next == Activate \/ Interrupt \/ (\E c \in Candidates, n \in Acknowledgments : Replay(c, n))
    \/ Match \/ Publish \/ Reject \/ Finish
Spec == Init /\ [][Next]_vars

PositionsOnce == \A i \in DOMAIN visible : visible[i].position = i
CheckedPrefix ==
    phase \in {"Reading", "Done"} =>
        /\ cursor <= Len(visible)
        /\ \A i \in 1..cursor : visible[i].value = output[i]
TerminalComplete == phase = "Done" => Len(visible) = Len(output)
CheckpointVerified ==
    phase \in {"Reading", "Done"} /\ checkpoint > 0 => acknowledged = checkpoint
ResumeUsesLatest == [][UseCheckpoint /\ attempt' > attempt => checkpoint' = Len(visible)]_vars
PrefixNeverChanges == [][
    Len(visible') >= Len(visible) /\ \A i \in DOMAIN visible : visible'[i] = visible[i]
]_vars
====
