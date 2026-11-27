---- MODULE PlaitPdStreamReplay ----
EXTENDS Naturals, Sequences

Decoders == {"old", "spare", "replacement"}
Outputs == {<<1, 2, 3>>, <<1, 4, 3>>, <<1>>}
NONE == "none"
Terminal == {"done", "failed", "timedOut"}
\* Three abstract clock ticks represent a finite request budget.
Budget == 3
VARIABLES routes, failed, decode, lease, phase, output, cursor, visible, remaining
vars == <<routes, failed, decode, lease, phase, output, cursor, visible, remaining>>

Init ==
    /\ routes = {"old", "spare"} /\ failed = {}
    /\ phase = "waiting" /\ decode = NONE /\ lease = NONE
    /\ cursor = 0 /\ output = <<>> /\ visible = <<>>
    /\ remaining = Budget

CanPoll == phase \notin Terminal /\ remaining > 0

Refresh ==
    /\ CanPoll /\ "replacement" \notin routes
    /\ routes' = routes \cup {"replacement"}
    /\ UNCHANGED <<failed, decode, lease, phase, output, cursor, visible, remaining>>

EmptyDirectory ==
    /\ CanPoll /\ routes # {}
    /\ routes' = {}
    /\ UNCHANGED <<failed, decode, lease, phase, output, cursor, visible, remaining>>

\* The same outer recovery deadline applies to every attempt. An already
\* admitted stream may still emit or finish after this budget reaches zero.
Tick ==
    /\ CanPoll /\ remaining' = remaining - 1
    /\ UNCHANGED <<routes, failed, decode, lease, phase, output, cursor, visible>>

ExpireWait ==
    /\ phase = "waiting" /\ remaining = 0
    /\ phase' = "timedOut"
    /\ UNCHANGED <<routes, failed, decode, lease, output, cursor, visible, remaining>>

Handoff(d, tokens) ==
    /\ phase = "waiting" /\ remaining > 0 /\ d \in routes \ failed
    /\ decode' = d /\ lease' = d /\ output' = tokens
    /\ phase' = "reading" /\ cursor' = 0
    /\ UNCHANGED <<routes, failed, visible, remaining>>

Disconnect ==
    /\ phase = "reading" /\ phase' = "waiting"
    /\ failed' = failed \cup {decode} /\ lease' = NONE
    /\ UNCHANGED <<routes, decode, output, cursor, visible, remaining>>

Match ==
    /\ phase = "reading" /\ cursor < Len(visible) /\ cursor < Len(output)
    /\ output[cursor+1] = visible[cursor+1].token /\ cursor' = cursor+1
    /\ UNCHANGED <<routes, failed, decode, lease, phase, output, visible, remaining>>

Publish ==
    /\ phase = "reading" /\ cursor = Len(visible) /\ cursor < Len(output)
    /\ cursor' = cursor+1
    /\ visible' = Append(visible, [position |-> cursor+1, token |-> output[cursor+1]])
    /\ UNCHANGED <<routes, failed, decode, lease, phase, output, remaining>>

Reject ==
    /\ phase = "reading" /\ cursor < Len(visible)
    /\ IF cursor = Len(output) THEN TRUE
       ELSE output[cursor+1] # visible[cursor+1].token
    /\ phase' = "failed" /\ lease' = NONE
    /\ UNCHANGED <<routes, failed, decode, output, cursor, visible, remaining>>

Finish ==
    /\ phase = "reading" /\ cursor = Len(output) /\ cursor = Len(visible)
    /\ phase' = "done" /\ lease' = NONE
    /\ UNCHANGED <<routes, failed, decode, output, cursor, visible, remaining>>

TerminalStay == phase \in Terminal /\ UNCHANGED vars

Next == Refresh \/ EmptyDirectory \/ Tick \/ ExpireWait \/ Disconnect \/ Match \/ Publish \/ Reject \/ Finish \/ TerminalStay
        \/ \E d \in Decoders, tokens \in Outputs : Handoff(d, tokens)
Spec == Init /\ [][Next]_vars /\ WF_vars(Tick) /\ WF_vars(ExpireWait)

BoundHandoff == IF phase = "reading" THEN lease = decode /\ decode \notin failed ELSE lease = NONE
PositionsOnce == \A i \in DOMAIN visible : visible[i].position = i
CheckedPrefix == phase \in {"reading", "done"} =>
    /\ cursor <= Len(visible)
    /\ \A i \in 1..cursor : visible[i].token = output[i]
TerminalComplete == phase = "done" => Len(visible) = Len(output)
TimeoutAtDeadline == phase = "timedOut" => remaining = 0
NoHandoffAfterDeadline == remaining = 0 =>
    \A d \in Decoders, tokens \in Outputs : ~ENABLED Handoff(d, tokens)
DeadlineNeverResets == [][remaining' <= remaining]_vars
WaitingEventuallyLeaves == phase = "waiting" ~> phase # "waiting"
====
