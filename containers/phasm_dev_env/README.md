
```bash
docker pull --platform linux/amd64 ubuntu:jammy
sudo docker build -f containers/phasm_dev_env/Dockerfile -t nbrei/phasm_dev_env:latest --platform linux/amd64 .
docker run -it -v `pwd`:/app --cap-add sys_ptrace --name phasm_dev nbrei/phasm_dev_env:latest
docker push nbrei/phasm_dev_env:latest
```
