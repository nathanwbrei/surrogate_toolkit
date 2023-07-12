
Building PHASM
==============

Acquiring a PHASM release
-------------------------

Requirements
------------
- C++17, gcc version, clang version. Mac, Linux?
- LLVM libraries
- Table of examples and their dependencies



------------


Building 
--------

PHASM can be built using 
.. code-block:: console

    $ git clone http://github.com/nathanwbrei/phasm
    $ cd phasm
    $ mkdir build
    $ cd build
    $ cmake .. -DCMAKE_PREFIX_PATH=/deps

