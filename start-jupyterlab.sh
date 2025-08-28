#!/bin/bash
# Launch Jupyter Lab on port 8888, killing any existing instances on other ports

# Check for existing Jupyter Lab processes
jlab_exist=$(ss -tulpn | grep jupyter-lab)
jlab_port=$(echo "$jlab_exist" | awk '{print $5}' | awk -F: '{print $NF}')

echo $jlab_exist
echo $jlab_port

# Kill existing instance if running on non-default port
if [[ -n $jlab_exist ]] && (( $jlab_port != 8888 )); then
   # Extract process ID from socket info
   jlab_pid=$(ss -tulpn 2>/dev/null | grep $jlab_port | grep -oP 'pid=\K[0-9]+')
   echo "Jupyter-lab is running but on the non-default port of $jlab_port. Restarting"
   echo $jlab_pid
   # Force kill existing process
   kill -9 $jlab_pid
   # Start Jupyter Lab on default port
   jupyter-lab --port 8888 --no-browser
else
   # Start Jupyter Lab on default port (no conflicts)
   jupyter-lab --port 8888 --no-browser
fi
