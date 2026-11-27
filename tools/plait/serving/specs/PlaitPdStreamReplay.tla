---- MODULE PlaitPdStreamReplay ----
EXTENDS Naturals, Sequences

Decoders == {"old", "spare", "replacement"}
Outputs == {<<1, 2, 3>>, <<1, 4, 3>>, <<1>>}
NONE == "none"
VARIABLES routes, failed, decode, lease, phase, output, cursor, visible
vars == <<routes, failed, decode, lease, phase, output, cursor, visible>>

Init ==
    /\ routes = {"old", "spare"} /\ failed = {}
    /\ phase = "waiting" /\ decode = NONE /\ lease = NONE
    /\ cursor = 0 /\ output = <<>> /\ visible = <<>>

Refresh ==
    /\ "replacement" \notin routes
    /\ routes' = routes \cup {"replacement"}
    /\ UNCHANGED <<failed, decode, lease, phase, output, cursor, visible>>

Handoff(d, tokens) ==
    /\ phase = "waiting" /\ d \in routes \ failed
    /\ decode' = d /\ lease' = d /\ output' = tokens
    /\ phase' = "reading" /\ cursor' = 0
    /\ UNCHANGED <<routes, failed, visible>>

Disconnect ==
    /\ phase = "reading" /\ phase' = "waiting"
    /\ failed' = failed \cup {decode} /\ lease' = NONE
    /\ UNCHANGED <<routes, decode, output, cursor, visible>>

Match ==
    /\ phase = "reading" /\ cursor < Len(visible) /\ cursor < Len(output)
    /\ output[cursor+1] = visible[cursor+1].token /\ cursor' = cursor+1
    /\ UNCHANGED <<routes, failed, decode, lease, phase, output, visible>>

Publish ==
    /\ phase = "reading" /\ cursor = Len(visible) /\ cursor < Len(output)
    /\ cursor' = cursor+1
    /\ visible' = Append(visible, [position |-> cursor+1, token |-> output[cursor+1]])
    /\ UNCHANGED <<routes, failed, decode, lease, phase, output>>

Reject ==
    /\ phase = "reading" /\ cursor < Len(visible)
    /\ IF cursor = Len(output) THEN TRUE
       ELSE output[cursor+1] # visible[cursor+1].token
    /\ phase' = "failed" /\ lease' = NONE
    /\ UNCHANGED <<routes, failed, decode, output, cursor, visible>>

Finish ==
    /\ phase = "reading" /\ cursor = Len(output) /\ cursor = Len(visible)
    /\ phase' = "done" /\ lease' = NONE
    /\ UNCHANGED <<routes, failed, decode, output, cursor, visible>>

Next == Refresh \/ Disconnect \/ Match \/ Publish \/ Reject \/ Finish
        \/ \E d \in Decoders, tokens \in Outputs : Handoff(d, tokens)
Spec == Init /\ [][Next]_vars

BoundHandoff == IF phase = "reading" THEN lease = decode /\ decode \notin failed ELSE lease = NONE
PositionsOnce == \A i \in DOMAIN visible : visible[i].position = i
CheckedPrefix == phase \in {"reading", "done"} =>
    /\ cursor <= Len(visible)
    /\ \A i \in 1..cursor : visible[i].token = output[i]
TerminalComplete == phase = "done" => Len(visible) = Len(output)
====
