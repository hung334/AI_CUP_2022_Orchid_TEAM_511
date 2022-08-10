Dim WshShell
Dim iURL 
Dim objShell
Set WshShell=WScript.CreateObject("WScript.Shell")
iURL = "http://127.0.0.1:8888"
WshShell.Run "cmd.exe"
WScript.Sleep 1500
WshShell.SendKeys "+%"
WshShell.SendKeys "ssh server0@140.118.170.220"
WshShell.SendKeys "{ENTER}"
WScript.Sleep 1500
WshShell.SendKeys "msplgodman"
WshShell.SendKeys "{ENTER}"
WScript.Sleep 1500
WshShell.SendKeys "cd Desktop/hung"
WshShell.SendKeys "{ENTER}"
WshShell.SendKeys "nohup jupyter lab --port 8888 --no-browser --NotebookApp.token='' &"
WshShell.SendKeys "{ENTER}"
WshShell.SendKeys "exit"
WshShell.SendKeys "{ENTER}"
WScript.Sleep 1500
WshShell.SendKeys "ssh -N -f -L localhost:8888:localhost:8888 server0@140.118.170.220"
WshShell.SendKeys "{ENTER}"
WScript.Sleep 1500
WshShell.SendKeys "msplgodman"
WshShell.SendKeys "{ENTER}"
set objShell = CreateObject("Shell.Application")
objShell.ShellExecute "chrome.exe", iURL, "", "", 1