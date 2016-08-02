import sys
import os
import paramiko


file_name = "Day_Sampler_Res.py"
path_to_upload = "Weiyi"
overwrite = False

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname="174.129.176.95", port=22, username="ubuntu", key_filename="/home/wlu/Desktop/rips")

if overwrite:
    upload_name = file_name
else:
    upload_name = "tmp.py"
sftp = client.open_sftp()
sftp.put(file_name, os.path.join("/home/ubuntu", path_to_upload, upload_name))
sftp.close()

cmd = "cd {};/home/ubuntu/anaconda2/bin/python2.7 {}".format(path_to_upload, upload_name)
stdin, stdout, stderr = client.exec_command(cmd)
while True:
    output = stdout.readline()
    if output == '':
        break
    if output:
        sys.stdout.write(output)
        sys.stdout.flush()

for line in stderr:
    sys.stdout.write(line)
    sys.stdout.flush()

if not overwrite:
    cmd = "cd {};rm {}".format(path_to_upload, upload_name)

client.close()
