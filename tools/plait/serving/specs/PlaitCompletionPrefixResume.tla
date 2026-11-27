---- MODULE PlaitCompletionPrefixResume ----
\* Native ingress, one greedy completion, without a connector. Each attempt
\* freezes the client-visible token and text prefix. A supporting engine
\* seeds its token history before recomputing context and sampling a suffix.
\* Token and text chunks can have different boundaries. Each validates the
\* entire corresponding prefix, and only the new suffix becomes visible.
EXTENDS PlaitCompletionReplay

VARIABLE shownText
prefixVars == <<active, attempt, output, cursor, visible, phase, checkpoint, acknowledged, shownText>>

Prompt == <<7, 8>>
SourceIds == [i \in DOMAIN Original |-> Original[i][1]] \o <<4>>
Limit == Len(SourceIds)
Behaviors == {"Supported", "Legacy", "NoAck", "WrongAck", "WrongText", "WrongToken", "WrongPrompt", "Short"}

Prefix(ids, text) == [ids |-> ids, text |-> text]
EmptyPrefix == Prefix(<<>>, <<>>)
Cursor(ids, text) == [ids |-> ids, text |-> text]
VisibleIds == [i \in DOMAIN visible |-> visible[i].value]
Request(ids) == [xargs |-> ids, kv |-> FALSE]
Engine(request, behavior, history, context, ids, text, ack, prompt) ==
    [request |-> request, behavior |-> behavior, history |-> history, context |-> context,
     ids |-> ids, text |-> text, ack |-> ack, prompt |-> prompt]

\* The target's natural prefix differs from the source. This fixture's next
\* token depends on the last token of the recomputed context.
Greedy(epoch, context, count) ==
    [i \in 1..count |-> 10 * epoch + context[Len(context)] + i]
TextFor(ids) == [i \in DOMAIN ids |-> 100 + ids[i]]
ReplaceFirst(values) == IF Len(values) = 0 THEN values ELSE [values EXCEPT ![1] = @ + 1]

Compute(epoch, request, behavior) ==
    LET history == IF behavior = "Legacy" THEN <<>> ELSE request.xargs
        prompt == IF behavior = "WrongPrompt" THEN <<9, 8>> ELSE Prompt
        context == prompt \o history
        predicted == history \o Greedy(epoch, context, Limit - Len(history))
        fullText == TextFor(predicted)
        short == behavior = "Short"
        wireIds == IF short THEN SubSeq(predicted, 1, Len(history) - 1)
                   ELSE IF behavior = "WrongToken" THEN ReplaceFirst(predicted) ELSE predicted
        wireText == IF short THEN SubSeq(fullText, 1, Len(history) - 1)
                    ELSE IF behavior = "WrongText" THEN ReplaceFirst(fullText) ELSE fullText
        ack == CASE behavior \in {"Legacy", "NoAck"} -> 0
                 [] behavior = "WrongAck" -> Len(history) + 1
                 [] OTHER -> Len(history)
    IN Engine(request, behavior, history, context, wireIds, wireText, ack, prompt)

InitPrefix ==
    /\ active = 1 /\ attempt = 1
    /\ output = Engine(Request(<<>>), "Source", <<>>, Prompt, SourceIds, TextFor(SourceIds), 0, Prompt)
    /\ cursor = Cursor(0, 0) /\ visible = <<>> /\ shownText = <<>>
    /\ phase = "Reading" /\ checkpoint = EmptyPrefix /\ acknowledged = 0

ActivatePrefix ==
    /\ active < 3 /\ active' = active + 1
    /\ UNCHANGED <<attempt, output, cursor, visible, phase, checkpoint, acknowledged, shownText>>

InterruptPrefix ==
    /\ phase \in {"Reading", "Starting"} /\ phase' = "Waiting"
    /\ UNCHANGED <<active, attempt, output, cursor, visible, checkpoint, acknowledged, shownText>>

StartPrefix ==
    /\ phase = "Waiting" /\ active > attempt
    /\ attempt' = active /\ checkpoint' = Prefix(VisibleIds, shownText)
    /\ output' = Engine(Request(VisibleIds), "Pending", <<>>, <<>>, <<>>, <<>>, 0, Prompt)
    /\ cursor' = Cursor(0, 0) /\ acknowledged' = 0 /\ phase' = "Starting"
    /\ UNCHANGED <<active, visible, shownText>>

Recompute(behavior) ==
    /\ phase = "Starting"
    /\ behavior = "Short" => Len(output.request.xargs) > 0
    /\ IF behavior # "Legacy" /\ Len(output.request.xargs) = Limit THEN
           /\ phase' = "Failed" /\ UNCHANGED output
       ELSE /\ output' = Compute(attempt, output.request, behavior)
            /\ phase' = "Reading"
    /\ UNCHANGED <<active, attempt, cursor, visible, checkpoint, acknowledged, shownText>>

Min(a, b) == IF a < b THEN a ELSE b
PrefixAt(prediction, expected, through) ==
    LET n == Min(Len(expected), through)
    IN SubSeq(prediction, 1, n) = SubSeq(expected, 1, n)
CompleteCheck(at) == at.ids >= Len(checkpoint.ids) /\ at.text >= Len(checkpoint.text)
AckIsRequired(count) == Len(checkpoint.ids) = 0 \/ count = Len(checkpoint.ids)

TailIds(through) ==
    [i \in 1..(through - Len(visible)) |->
        [position |-> Len(visible) + i, value |-> output.ids[Len(visible) + i]]]

ReceivePrefix(ids, text) ==
    /\ phase = "Reading" /\ ids + text > 0
    /\ LET at == Cursor(cursor.ids + ids, cursor.text + text)
           count == IF cursor = Cursor(0, 0) THEN output.ack ELSE acknowledged
       IN /\ at.ids <= Len(output.ids) /\ at.text <= Len(output.text)
          /\ IF AckIsRequired(count) /\ output.prompt = Prompt /\
                PrefixAt(output.ids, checkpoint.ids, at.ids) /\ PrefixAt(output.text, checkpoint.text, at.text)
             THEN /\ cursor' = at /\ acknowledged' = count /\ UNCHANGED phase
                  /\ IF CompleteCheck(at) THEN
                         /\ visible' = visible \o TailIds(at.ids)
                         /\ shownText' = shownText \o SubSeq(output.text, Len(shownText) + 1, at.text)
                     ELSE UNCHANGED <<visible, shownText>>
             ELSE /\ phase' = "Failed" /\ UNCHANGED <<cursor, visible, shownText, acknowledged>>
    /\ UNCHANGED <<active, attempt, output, checkpoint>>

TerminalPrefix ==
    /\ phase = "Reading" /\ cursor.ids = Len(output.ids) /\ cursor.text = Len(output.text)
    /\ phase' = IF CompleteCheck(cursor) /\ AckIsRequired(acknowledged) /\ output.prompt = Prompt
                THEN "Done" ELSE "Failed"
    /\ UNCHANGED <<active, attempt, output, cursor, visible, checkpoint, acknowledged, shownText>>

NextPrefix == ActivatePrefix \/ InterruptPrefix \/ StartPrefix \/ TerminalPrefix
    \/ (\E behavior \in Behaviors : Recompute(behavior))
    \/ (\E ids, text \in 0..2 : ReceivePrefix(ids, text))

PrefixSpec == InitPrefix /\ [][NextPrefix]_prefixVars

PromptHistoryBeforeDecode ==
    /\ ~output.request.kv /\ output.request.xargs = checkpoint.ids
    /\ output.behavior = "Supported" =>
           /\ Len(checkpoint.ids) < Limit
           /\ output.history = checkpoint.ids /\ output.context = Prompt \o checkpoint.ids
           /\ output.ids = checkpoint.ids \o Greedy(attempt, output.context, Limit - Len(checkpoint.ids))

FirstChoiceAcknowledged ==
    phase \in {"Reading", "Done"} /\ cursor # Cursor(0, 0) =>
        /\ output.prompt = Prompt
        /\ Len(checkpoint.ids) > 0 => acknowledged = Len(checkpoint.ids)

OnlyVerifiedSuffixVisible ==
    phase \in {"Reading", "Done"} =>
        IF Len(visible) > Len(checkpoint.ids) \/ Len(shownText) > Len(checkpoint.text)
        THEN /\ cursor.ids >= Len(checkpoint.ids) /\ cursor.text >= Len(checkpoint.text)
             /\ VisibleIds = SubSeq(output.ids, 1, cursor.ids)
             /\ shownText = SubSeq(output.text, 1, cursor.text)
        ELSE /\ VisibleIds = checkpoint.ids /\ shownText = checkpoint.text

TerminalPrefixWasVerified ==
    phase = "Done" =>
        /\ Len(checkpoint.ids) <= cursor.ids /\ Len(checkpoint.text) <= cursor.text
        /\ VisibleIds = output.ids /\ shownText = output.text

TakeoverFreezesLatestVisible == [][attempt' > attempt =>
    checkpoint' = Prefix(VisibleIds, shownText) /\ output'.request = Request(VisibleIds)]_prefixVars

TextNeverChanges == [][
    Len(shownText') >= Len(shownText) /\ \A i \in DOMAIN shownText : shownText'[i] = shownText[i]
]_prefixVars
====
