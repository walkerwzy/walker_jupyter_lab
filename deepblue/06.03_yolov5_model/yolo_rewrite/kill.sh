#!/bin/bash
ps -ef | grep YOLO_TOKEN_AAA
ps -ef | grep YOLO_TOKEN_AAA | awk '{print $2}' | xargs kill -9