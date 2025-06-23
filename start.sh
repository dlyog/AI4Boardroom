#!/bin/bash
nohup python app.py > output.log 2>&1 &
echo $! > pid.file