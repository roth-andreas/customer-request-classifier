#!/bin/bash

/bin/ollama serve &

pid=$!

sleep 5

echo "Retrieve Llama3 model"
ollama pull llama3
echo "Done!"

wait $pid
