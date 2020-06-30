#!/bin/bash

rm -r logs
mkdir logs
tensorboard --logdir logs
