
# ✅ **Operating System: Complete Topic-wise Checklist**

---

## 📘 **1. Introduction & Basics**

* What is an Operating System
* Functions and Goals of OS
* Types of Operating Systems:

  * Batch OS
  * Time-Sharing OS
  * Distributed OS
  * Real-Time OS (Hard & Soft)
  * Embedded OS
  * Network OS
  * Mobile OS
* OS Structures:

  * Monolithic
  * Layered
  * Microkernel
  * Modular
  * Hybrid
* System Calls vs API
* OS Services
* User Mode vs Kernel Mode
* Boot Process:

  * BIOS/UEFI
  * Bootloader
  * Kernel Initialization

---

## 🧑‍💻 **2. Process Management**

* Process Concepts:

  * PCB (Process Control Block)
  * Process States (New, Ready, Running, Waiting, Terminated)
  * Context Switching
* Types of Processes:

  * User vs System
  * Foreground vs Background
  * Independent vs Cooperating
* Process Scheduling:

  * Scheduling Queues
  * Dispatcher
  * Preemptive vs Non-Preemptive

### 🔁 Scheduling Algorithms:

* FCFS (First-Come, First-Served)
* SJF (Shortest Job First)
* SRTF (Shortest Remaining Time First)
* Priority Scheduling (Preemptive & Non-Preemptive)
* Round Robin (RR)
* Multilevel Queue Scheduling
* Multilevel Feedback Queue
* Lottery Scheduling

---

## 🧵 **3. Thread Management**

* Threads vs Processes
* User-Level vs Kernel-Level Threads
* Thread Libraries:

  * POSIX (pthreads)
  * Windows Threads
  * Java Threads
* Multithreading Models:

  * Many-to-One
  * One-to-One
  * Many-to-Many
* Thread Pooling
* Thread Synchronization
* Thread Scheduling

---

## 🧷 **4. Inter-Process Communication (IPC)**

* Shared Memory
* Message Passing
* Pipes (Named & Unnamed)
* Sockets
* Signals

---

## ⚔️ **5. Process Synchronization**

* Critical Section Problem
* Race Condition
* Synchronization Tools:

  * Mutex
  * Binary & Counting Semaphores
  * Spinlocks
  * Monitors
  * Condition Variables

### 🔐 Classical Problems:

* Producer-Consumer Problem (Bounded Buffer)
* Readers-Writers Problem
* Dining Philosophers Problem
* Sleeping Barber Problem

### 🧮 Algorithms:

* Peterson’s Algorithm
* Lamport’s Bakery Algorithm
* Test-and-Set
* Compare-and-Swap

---

## 🔁 **6. Deadlocks**

* Necessary Conditions (Coffman Conditions)
* Resource Allocation Graph
* Deadlock Prevention
* Deadlock Avoidance:

  * Safe State
  * Banker’s Algorithm
* Deadlock Detection:

  * Wait-for Graph
* Deadlock Recovery
* Starvation vs Deadlock

---

## 🧠 **7. Memory Management**

* Address Binding (Compile-time, Load-time, Execution-time)
* Logical vs Physical Address
* MMU (Memory Management Unit)
* Swapping
* Contiguous Allocation:

  * Fixed Partitioning
  * Variable Partitioning
* Fragmentation:

  * Internal
  * External
* Compaction

### 📦 Paging:

* Page Tables
* Inverted Page Table
* TLB (Translation Lookaside Buffer)
* Multi-level Paging

### 🧩 Segmentation:

* Segment Table
* Segmentation + Paging (Hybrid)

---

## 🔮 **8. Virtual Memory**

* Concept of Virtual Memory
* Demand Paging
* Copy-on-Write (COW)
* Page Faults and Handling
* Memory-Mapped Files
* Thrashing
* Working Set Model
* Page Fault Frequency
* Belady’s Anomaly

### 🔄 Page Replacement Algorithms:

* FIFO
* LRU (Least Recently Used)
* Optimal
* LFU (Least Frequently Used)
* MFU (Most Frequently Used)
* Second-Chance (Clock)

### 🔢 Allocation Algorithms:

* Equal Allocation
* Proportional Allocation
* Global vs Local Allocation

---

## 🗂️ **9. File System**

* File Attributes
* File Types and Structures
* File Operations
* File Access Methods:

  * Sequential
  * Direct
  * Indexed
* File System Mounting
* Directory Structures:

  * Single-level
  * Two-level
  * Tree-structured
  * Acyclic Graph
  * General Graph

### 📦 File Allocation Methods:

* Contiguous
* Linked
* Indexed (Single, Multilevel, Combined)

### 🧮 Free Space Management:

* Bit Vector
* Linked List
* Grouping
* Counting

### 📊 File System Implementation:

* Inodes
* Superblock
* Journaling File Systems
* Virtual File Systems (VFS)

---

## 💽 **10. Disk Management & I/O**

* Disk Structure:

  * Platters, Tracks, Sectors
  * SSDs vs HDDs
* Disk Formatting
* Bad Block Recovery

### 📅 Disk Scheduling Algorithms:

* FCFS
* SSTF (Shortest Seek Time First)
* SCAN (Elevator)
* C-SCAN
* LOOK
* C-LOOK

### ⚙️ RAID Levels:

* RAID 0, 1, 5, 6, 10

---

## 🔌 **11. I/O Systems**

* I/O Hardware: Interrupts, Polling, DMA
* Kernel I/O Subsystems:

  * Buffering
  * Caching
  * Spooling
* Device Drivers
* I/O Scheduling
* I/O Ports vs Memory-Mapped I/O

---

## 🔒 **12. Protection and Security**

* Protection:

  * Access Matrix
  * ACLs (Access Control Lists)
  * Capability Lists

* Security Goals:

  * Confidentiality
  * Integrity
  * Availability

### 🛡️ Topics:

* Authentication vs Authorization
* User & Group Management
* Threats:

  * Malware (Viruses, Worms, Trojans)
  * Denial of Service (DoS)
  * Man-in-the-Middle
* Encryption:

  * Symmetric
  * Asymmetric
* Intrusion Detection Systems (IDS)
* Firewalls

---

## 🛰️ **13. Distributed Systems (Basics)**

* Characteristics and Examples
* Network OS vs Distributed OS
* Distributed Coordination
* RPC (Remote Procedure Call)
* Clock Synchronization:

  * NTP
  * Lamport Timestamps
* Distributed Mutual Exclusion
* Election Algorithms:

  * Bully Algorithm
  * Ring Algorithm
* Distributed Deadlock Detection
* Replication

---

## 🕰️ **14. Real-Time Systems**

* Hard vs Soft Real-Time
* Real-Time Scheduling:

  * Rate Monotonic Scheduling
  * Earliest Deadline First (EDF)
* Priority Inversion
* Priority Inheritance & Ceiling Protocol

---

## 📱 **15. Mobile & Embedded OS**

* Resource Management in Mobile OS
* Power Management
* Memory Constraints
* Real-Time Constraints in Embedded Systems
* Embedded Kernel Design

---

## 🧪 **16. System Calls & APIs**

* System Call Interface
* Types of System Calls:

  * Process Control
  * File Manipulation
  * Device Management
  * Information Maintenance
* Common UNIX/Linux Calls:

  * `fork()`, `exec()`, `wait()`, `exit()`, `kill()`

---

## 🖥️ **17. Virtualization & Cloud OS**

* Virtual Machines (VMs)
* Hypervisors:

  * Type 1 (Bare Metal)
  * Type 2 (Hosted)
* Containers (Docker Basics)
* Cloud OS Concepts

---

## 📊 **18. Performance & Optimization**

* Bottleneck Analysis
* System Tuning
* Load Balancing
* Caching Strategies
* CPU, I/O, and Memory Optimization

---

## 🧑‍🔬 **19. OS Case Studies**

* Linux Architecture
* Windows Internals
* Android OS Overview
* UNIX Philosophy

---

Let me know if you want:

* ✅ A **1-week study plan**
* 🧠 Topic-wise **interview questions**
* 📌 Important **mnemonics or mind maps**
* 📄 Printable **PDF of this list**

I’m happy to help you revise smartly!
