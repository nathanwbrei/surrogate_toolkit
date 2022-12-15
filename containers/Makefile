

image:
	-docker rm nbrei_surrogate_image
	docker build -t nbrei_surrogate_image .

image-nc:
	-docker rm nbrei_surrogate_image
	docker build --no-cache -t nbrei_surrogate_image .

run:
	-docker rm nbrei_surrogate_container
	ssh-keygen -f "${HOME}/.ssh/known_hosts" -R "[localhost]:2222"
	docker run -d \
			   -v /Users/nbrei/projects/surrogate/surrogate_data:/data \
	           -p 2222:22 \
			   --cap-add sys_ptrace \
		       --name nbrei_surrogate_container \
		       nbrei_surrogate_image /sbin/sshd -D

shell:
	docker exec -it nbrei_surrogate_container /bin/bash

ssh:
	ssh -p 2222 root@localhost


