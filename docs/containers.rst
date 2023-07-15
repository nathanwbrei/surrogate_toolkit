
Containers
==========

Docker containers
-----------------

There are three docker containers for working with PHASM. 

`phasm_dev_env` is based off of Ubuntu 20.04 Jammy, and provides the build environment, 
plus all of the dependencies necessary for building and running the examples. It is 5.6GB.

`phasm_cuda_env` is based off of the NVIDIA's Ubuntu 20.04 Jammy CUDA container. It provides
the same build environment and dependencies as `phasm_dev_env`. This is intended for 
running PHASM on real heterogeneous hardware, e.g. for benchmarking or 'prod'. It is 6.6GB.

`phasm_mini_env` is based off of Ubuntu 20.04 Jammy, and provides the build environment stripped 
of the dependencies needed to run the examples. TODO: Push this to Dockerhub


On DockerHub
~~~~~~~~~~~~
They can be downloaded straight from DockerHub:

.. code-block:: console
    
    $ docker pull nbrei/phasm_dev_env:latest
    $ docker pull nbrei/phasm_cuda_env:latest
    $ docker pull nbrei/phasm_mini_env:latest


Building locally
~~~~~~~~~~~~~~~~
Building the PHASM Docker containers locally takes a long time and often fails due to 
out-of-disk-space errors. However, this can be useful for omitting heavy dependencies 
such as Geant4 or Julia. Note that all dependencies are downloaded before any Docker commands 
are run. (This solves some problems involving Docker re-downloading multi-GB files, and 
also avoids problems with SSL certificates when on JLab's network.)
From the PHASM source code root directory, run:

.. code-block:: console

    $ export PHASM_USE_GEANT4=0
    $ scripts/download_deps.sh 
    $ sudo docker build -t phasm_custom_env -f containers/phasm_dev_env/Dockerfile
                        --build-arg PHASM_USE_GEANT4=0



Running locally
~~~~~~~~~~~~~~~
Once you have the Docker container, all you have to do to access the PHASM environment is
to mount the host volume containing PHASM (and whatever codebase PHASM is interacting with).
Assuming your source code lives at `~/phasm`, run the following:

.. code-block:: console

    docker run -it -v ~/phasm:/app nbrei/phasm_dev_env:latest 


Singularity/Apptainer containers
--------------------------------

Obtaining a prebuilt SIF file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PHASM singularity containers are available on JLab's shared work disk. You can 
immediately run it as follows:

.. code-block:: console

    singularity shell -B /u,/group,/w,/work /work/epsci/shared_pkg/phasm_dev_env.sif


TODO: The PHASM singularity containers will shortly become available on CVMFS. 


Building locally
~~~~~~~~~~~~~~~~

If you don't have access to JLab's shared work disk, you can build a singularity container 
immediately from the corresponding Docker container. Note that Singularity needs a lot of disk space
for this. We strongly recommend setting both `SINGULARITY_TMPDIR` and `SINGULARITY_CACHEDIR` to a path 
with at least 20GB of free space. 

.. code-block:: console

    mkdir -p /scratch/$USER/singularity
    export SINGULARITY_TMPDIR=/scratch/$USER/singularity
    export SINGULARITY_CACHEDIR=/scratch/$USER/singularity
    singularity build phasm_dev_env.sif docker://nbrei/phasm_dev_env:latest


It may happen that you wish to build a singularity container on a Mac;
for this purpose we have included a Docker container that hosts singularity. 

.. code-block:: console
    
    host% docker build -t singularity_env -f containers/singularity_env/Dockerfile
    host% docker run -it -v $PATH_TO_PHASM_SOURCE:/app,$PATH_TO_LARGE_SCRATCH_DISK:/scratch singularity_env 

    docker% export SINGULARITY_TMPDIR=/scratch
    docker% export SINGULARITY_CACHEDIR=/scratch
    docker% singularity build phasm_custom_env.sif docker://nbrei/phasm_dev_env:latest


Connecting to an IDE
~~~~~~~~~~~~~~~~~~~~

If your IDE supports it, the best way to connect your IDE to the PHASM singularity container
is via SSH using the `RemoteCommand` feature. This has been tested with Visual Studio Code.
This even works on your local machine! In your `~/.ssh/config` file, add something like:

.. code-block:: text

    Host my_phasm_env
        RemoteCommand singularity shell -B PATH_TO_PHASM:/app phasm_dev_env.sif


If you are on JLab's ifarm, the full configuration is as follows:

.. code-block:: text

    Host ifarm_phasm
       HostName ifarm1901.jlab.org
       ProxyJump scilogin.jlab.org
       RemoteCommand singularity shell --bind /u,/group,/w,/work /work/epsci/shared_pkg/phasm_dev_env.sif
       RequestTTY yes
       ForwardX11 yes

For more information, see `the article on EPSCIwiki <https://wiki.jlab.org/epsciwiki/index.php/Jupyter_via_VSCode_remote-ssh_with_singularity_on_ifarm>`_.

Submitting to SLURM
~~~~~~~~~~~~~~~~~~~
TODO: Update me
See farm_guide_singularity.md for slightly out-of-date instructions.