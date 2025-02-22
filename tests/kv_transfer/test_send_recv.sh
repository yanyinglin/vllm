#!/bin/bash

RANK=0 python3 test_send_recv.py &
PID0=$!
RANK=1 python3 test_send_recv.py &
PID1=$!

wait $PID0
wait $PID1

# RANK=0 python3 /app/share/mixflow/entry/test_send_recv.py
# RANK=1 python3 /app/share/mixflow/entry/test_send_recv.py