---- MODULE PlaitShardedPromptScores ----
EXTENDS Integers, FiniteSets, Sequences

\* Two TP projection slots of width five, with only two valid cells in
\* the second slot. At k = 2 its candidate packet also needs padding.
\* Scores are exact integers. Nonfinite input is flagged separately and
\* uses the existing dense scorer. Floating-point normalization is outside
\* this model. Targets are valid IDs and all slots use the same batch and k.
Slots == 0..1
Width == 5
Tokens == 0..6
Columns == 0..(Width - 1)
Rows == 1..2
ScoreValues == {-1, 0, 1}
TopCounts == 0..2

Owner(token) == token \div Width
LocalTokens(slot) == {token \in Tokens : Owner(token) = slot}
Row(scores, target, badSlots) ==
    [scores |-> scores, target |-> target, badSlots |-> badSlots]

Project(scores, slot) ==
    [c \in Columns |-> LET token == slot * Width + c IN
        IF token \in Tokens THEN scores[token] ELSE 99]

DenseVocab(scores) ==
    LET shards == [slot \in Slots |-> Project(scores, slot)]
    IN [token \in Tokens |-> shards[Owner(token)][token % Width]]

Before(scores, a, b, ascending) ==
    \/ scores[a] > scores[b]
    \/ /\ scores[a] = scores[b]
       /\ IF ascending THEN a < b ELSE a > b

\* Dense and local top-k use different tie orders here. A fallback row
\* uses the dense scorer's convention. Its concrete tie-ID stability
\* across batch shapes depends on the dense implementation.
TopIds(scores, candidates, count, ascending) ==
    [i \in 1..count |->
        CHOOSE token \in candidates :
            Cardinality({other \in candidates : Before(scores, other, token, ascending)}) = i - 1]

ScoresAt(scores, ids) == [i \in DOMAIN ids |-> scores[ids[i]]]
ExtraCount(k) == IF k = 0 THEN 0 ELSE k + 1
Limit(candidates, count) == IF Cardinality(candidates) < count THEN Cardinality(candidates) ELSE count

SelectedInShard(row, slot) ==
    LET relative == row.target - slot * Width
        last == Cardinality(LocalTokens(slot)) - 1
        column == IF relative < 0 THEN 0 ELSE IF relative > last THEN last ELSE relative
    IN Project(row.scores, slot)[column]

CandidatesInShard(scores, slot, k) ==
    LET tokens == LocalTokens(slot)
        projected == Project(scores, slot)
        local == [t \in tokens |-> projected[t % Width]]
        ids == TopIds(local, tokens, Limit(tokens, ExtraCount(k)), FALSE)
    IN [token \in {ids[i] : i \in DOMAIN ids} |-> local[token]]

\* The numeric checks omit candidate placeholders at -Inf. Each slot still
\* sends a fixed-width packet. A valid finite candidate always beats padding.
Packet(row, slot, k) ==
    [selected |-> SelectedInShard(row, slot),
     finite |-> slot \notin row.badSlots,
     candidates |-> CandidatesInShard(row.scores, slot, k)]

MergeCandidates(packets) ==
    LET candidates == UNION {DOMAIN packets[slot].candidates : slot \in Slots}
    IN [t \in candidates |-> packets[Owner(t)].candidates[t]]

MergedTop(packets, k) ==
    LET candidates == MergeCandidates(packets)
    IN TopIds(candidates, DOMAIN candidates, ExtraCount(k), FALSE)

HasTie(scores) ==
    \E i \in 1..(Len(scores) - 1) : scores[i] = scores[i + 1]

NeedsFallback(packets, k) ==
    \/ \E slot \in Slots : ~packets[slot].finite
    \/ HasTie(ScoresAt(MergeCandidates(packets), MergedTop(packets, k)))

DenseNeedsFallback(row, k) ==
    LET scores == DenseVocab(row.scores)
        ids == TopIds(scores, Tokens, k, TRUE)
    IN \/ row.badSlots # {}
       \/ \E i \in 1..k :
            Cardinality({t \in Tokens : scores[t] = scores[ids[i]]}) > 1

CountAtLeast(scores, threshold) ==
    Cardinality({t \in DOMAIN scores : scores[t] >= threshold})

DenseRank(scores, threshold) ==
    Cardinality(Tokens) - Cardinality({t \in Tokens : scores[t] < threshold})

RankPart(scores, slot, threshold) ==
    LET projected == Project(scores, slot)
    IN CountAtLeast([t \in LocalTokens(slot) |-> projected[t % Width]], threshold)

RankSum(scores, threshold) == RankPart(scores, 0, threshold) + RankPart(scores, 1, threshold)

Result(pick, score, ids, values, rank) ==
    [ids |-> <<pick>> \o ids, scores |-> <<score>> \o values, rank |-> rank]

DenseResult(row, k) ==
    LET dense == DenseVocab(row.scores)
        ids == TopIds(dense, Tokens, k, TRUE)
        threshold == dense[row.target]
    IN Result(row.target, threshold, ids, ScoresAt(dense, ids), DenseRank(dense, threshold))

ShardedResult(row, packets, k) ==
    IF NeedsFallback(packets, k) THEN DenseResult(row, k)
    ELSE
        LET ids == SubSeq(MergedTop(packets, k), 1, k)
            selected == packets[Owner(row.target)].selected
        IN Result(row.target, selected, ids, ScoresAt(MergeCandidates(packets), ids),
                  RankSum(row.scores, selected))

RowRefines(scores) ==
    /\ DenseVocab(scores) = scores
    /\ \A target \in Tokens :
        LET row == Row(scores, target, {})
            selected == SelectedInShard(row, Owner(target))
        IN /\ selected = scores[target]
           /\ RankSum(scores, selected) = DenseRank(scores, scores[target])
    /\ \A count \in TopCounts :
        LET row == Row(scores, 0, {})
            packets == [slot \in Slots |-> Packet(row, slot, count)]
            candidates == MergeCandidates(packets)
            globalIds == TopIds(scores, Tokens, ExtraCount(count), TRUE)
            mergedIds == MergedTop(packets, count)
        IN /\ ScoresAt(candidates, mergedIds) = ScoresAt(scores, globalIds)
           /\ NeedsFallback(packets, count) = DenseNeedsFallback(row, count)
           /\ ~NeedsFallback(packets, count) =>
                /\ \A i \in 1..count : globalIds[i] \in DOMAIN candidates
                /\ SubSeq(mergedIds, 1, count) = SubSeq(globalIds, 1, count)

\* Batches cover ties within top-k, at its boundary, and below the cutoff.
\* Bad slots also exercise both fallback branches at k = 0.
Fixture(values, target, badSlots) ==
    Row([t \in Tokens |-> values[t + 1]], target, badSlots)

Cases ==
    {Fixture(<<1, 0, -1, -1, -1, -1, -1>>, 0, {}),
     Fixture(<<1, 1, 0, -1, -1, -1, -1>>, 3, {}),
     Fixture(<<1, 0, -1, -1, -1, 0, -1>>, 5, {}),
     Fixture(<<1, 0, -1, -1, -1, -1, -1>>, 6, {1})}

VARIABLES batch, k, logitsMode, received, outstanding, completed
vars == <<batch, k, logitsMode, received, outstanding, completed>>

NoCall == [kind |-> "None", rows |-> {}, width |-> 0, axis |-> 0]
Call(kind, rows, width, axis) ==
    [kind |-> kind, rows |-> rows, width |-> width, axis |-> axis]

MetadataCall == Call("Metadata", Rows, (IF logitsMode THEN 2 ELSE 4) + 2 * ExtraCount(k), 0)
RankCall == Call("RankSum", Rows, 1, 0)
DenseCall(rows) == Call("Dense", rows, Width, -1)

\* The all-gather delivers identical packet order (slot, then row) at every
\* participant. Invocation arrival and the choice of what to invoke are local.
PacketsToSend == [slot \in Slots |-> [row \in Rows |-> Packet(batch[row], slot, k)]]
Gathered(observer, row) == [slot \in Slots |-> received[observer][slot][row]]
FallbackRows(observer) == {row \in Rows : NeedsFallback(Gathered(observer, row), k)}
DenseFallbackRows == {row \in Rows : DenseNeedsFallback(batch[row], k)}

NextCall(observer) ==
    IF Len(completed) = 0 THEN MetadataCall
    ELSE LET fallback == FallbackRows(observer) IN
        CASE Len(completed) = 1 /\ fallback = Rows -> DenseCall(Rows)
          [] Len(completed) = 1 /\ fallback # Rows -> RankCall
          [] Len(completed) = 2 /\ fallback # {} /\ fallback # Rows -> DenseCall(fallback)
          [] OTHER -> NoCall

ExpectedCalls ==
    IF DenseFallbackRows = Rows THEN <<MetadataCall, DenseCall(Rows)>>
    ELSE <<MetadataCall, RankCall>> \o
         (IF DenseFallbackRows = {} THEN <<>> ELSE <<DenseCall(DenseFallbackRows)>>)

EmptyTransport ==
    /\ received = [slot \in Slots |-> <<>>]
    /\ outstanding = [slot \in Slots |-> NoCall]
    /\ completed = <<>>

Init ==
    /\ batch \in [Rows -> Cases]
    /\ k \in TopCounts
    /\ logitsMode \in BOOLEAN
    /\ EmptyTransport

\* Exhaust all 3^7 score vectors. Keep this as a separate state space so
\* the numeric oracle runs once per vector rather than on each batch step.
InitRowCheck ==
    /\ \E scores \in [Tokens -> ScoreValues] :
        batch = [row \in Rows |-> Row(scores, 0, {})]
    /\ k = 0
    /\ logitsMode = TRUE
    /\ EmptyTransport

NextRowCheck == UNCHANGED vars
FiniteRowRefinement == RowRefines(batch[1].scores)

Issue(slot) ==
    /\ outstanding[slot] = NoCall
    /\ LET call == NextCall(slot) IN
        /\ call # NoCall
        /\ outstanding' = [outstanding EXCEPT ![slot] = call]
    /\ UNCHANGED <<batch, k, logitsMode, received, completed>>

Finish ==
    /\ \E slot \in Slots : outstanding[slot] # NoCall
    /\ \A a, b \in Slots : outstanding[a] = outstanding[b]
    /\ completed' = Append(completed, outstanding[0])
    /\ received' = IF outstanding[0] = MetadataCall
                  THEN [slot \in Slots |-> PacketsToSend] ELSE received
    /\ outstanding' = [slot \in Slots |-> NoCall]
    /\ UNCHANGED <<batch, k, logitsMode>>

Next == Finish \/ \E slot \in Slots : Issue(slot)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

FallbackAgreement ==
    Len(completed) > 0 =>
        \A observer \in Slots : FallbackRows(observer) = DenseFallbackRows

CollectiveOrder ==
    /\ Len(completed) <= Len(ExpectedCalls)
    /\ completed = SubSeq(ExpectedCalls, 1, Len(completed))
    /\ \A slot \in Slots :
        outstanding[slot] # NoCall =>
            /\ Len(completed) < Len(ExpectedCalls)
            /\ outstanding[slot] = ExpectedCalls[Len(completed) + 1]

ScoresAgree ==
    completed = ExpectedCalls =>
        \A slot \in Slots, row \in Rows :
            ShardedResult(batch[row], Gathered(slot, row), k) = DenseResult(batch[row], k)

AllCallsFinish == <> (completed = ExpectedCalls)
====
