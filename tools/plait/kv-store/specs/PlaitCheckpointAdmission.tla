---- MODULE PlaitCheckpointAdmission ----
EXTENDS Integers

\* Optional writes charge credit before sending. A read reserves a whole
\* page before separate workers claim portions. Both hold their charge
\* until completion or cancellation. One tick earns fresh byte credit.
Burst == 4
Rate == 1
MaxTime == 7
VARIABLES time, credit, leased, reading, writing, sent
vars == <<time, credit, leased, reading, writing, sent>>
Pending == leased + reading + writing
Min(a, b) == IF a <= b THEN a ELSE b

Init ==
    /\ time = 0
    /\ credit = Burst
    /\ leased = 0
    /\ reading = 0
    /\ writing = 0
    /\ sent = 0

Tick ==
    /\ time < MaxTime
    /\ time' = time + 1
    /\ credit' = Min(credit + Rate, Burst - Pending)
    /\ UNCHANGED <<leased, reading, writing, sent>>

ReserveRead(bytes) ==
    /\ credit >= bytes
    /\ credit' = credit - bytes
    /\ leased' = leased + bytes
    /\ UNCHANGED <<time, reading, writing, sent>>

ClaimRead(bytes) ==
    /\ leased >= bytes
    /\ leased' = leased - bytes
    /\ reading' = reading + bytes
    /\ UNCHANGED <<time, credit, writing, sent>>

AdmitWrite(bytes) ==
    /\ credit >= bytes
    /\ credit' = credit - bytes
    /\ writing' = writing + bytes
    /\ UNCHANGED <<time, leased, reading, sent>>

FinishRead(bytes, complete) ==
    /\ reading >= bytes
    /\ reading' = reading - bytes
    /\ sent' = sent + (IF complete THEN bytes ELSE 0)
    /\ UNCHANGED <<time, credit, leased, writing>>

FinishWrite(bytes, complete) ==
    /\ writing >= bytes
    /\ writing' = writing - bytes
    /\ sent' = sent + (IF complete THEN bytes ELSE 0)
    /\ UNCHANGED <<time, credit, leased, reading>>

Expire(bytes) ==
    /\ leased >= bytes
    /\ leased' = leased - bytes
    /\ UNCHANGED <<time, credit, reading, writing, sent>>

Next == Tick \/ (\E bytes \in 1..Burst :
    ReserveRead(bytes) \/ ClaimRead(bytes) \/ AdmitWrite(bytes) \/ Expire(bytes)
    \/ (\E complete \in BOOLEAN : FinishRead(bytes, complete) \/ FinishWrite(bytes, complete)))

Spec == Init /\ [][Next]_vars
TypeOK == /\ time \in 0..MaxTime
          /\ credit \in 0..Burst /\ leased \in 0..Burst
          /\ reading \in 0..Burst /\ writing \in 0..Burst
          /\ sent \in 0..(Burst + Rate * MaxTime)
SharedBurst == credit + Pending <= Burst
ByteAllowance == sent + credit + Pending <= Burst + Rate * time
====
