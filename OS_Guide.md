
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

Absolutely, Renu! Let's begin with **📘 1. Introduction & Basics** of Operating Systems (OS), covering each topic in **detailed explanation** followed by **interview-style Q\&A** to reinforce your understanding and help you revise effectively.

---

# 📘 1. Introduction & Basics – Detailed Explanation

---

### ✅ **What is an Operating System?**

An **Operating System (OS)** is **system software** that acts as an **intermediary between users and computer hardware**. It provides a user-friendly interface and manages all hardware and software resources on a computer.

#### Functions:

* Controls and allocates memory
* Schedules tasks and processes
* Manages files and I/O devices
* Provides security and access control

---

### ✅ **Functions and Goals of OS**

**Primary Functions:**

1. **Process Management:** Creating, scheduling, terminating processes.
2. **Memory Management:** Allocating/deallocating memory space.
3. **File System Management:** Organizing, storing, and retrieving files.
4. **Device Management:** Controlling peripherals using device drivers.
5. **Security and Protection:** Preventing unauthorized access.
6. **User Interface:** CLI or GUI for user interaction.

**Goals:**

* **Convenience** (user-friendliness)
* **Efficiency** (maximize resource utilization)
* **Ability to evolve** (adapt to new hardware and features)

---

### ✅ **Types of Operating Systems**

| Type                | Description                                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Batch OS**        | Executes batches of jobs with no interaction. Jobs are processed in bulk.                                                                                    |
| **Time-Sharing OS** | Allows multiple users to share system resources simultaneously. Uses CPU scheduling and multi-programming.                                                   |
| **Distributed OS**  | Manages a group of independent computers to appear as a single system.                                                                                       |
| **Real-Time OS**    | Ensures tasks are executed within strict timing constraints. <br> 🔸 *Hard Real-Time:* Guarantees timing. <br> 🔸 *Soft Real-Time:* Tries to meet deadlines. |
| **Embedded OS**     | Designed to operate embedded systems (e.g., washing machines, IoT devices). Limited functionality.                                                           |
| **Network OS**      | Provides services to computers connected via a network (e.g., Novell NetWare).                                                                               |
| **Mobile OS**       | Optimized for mobile devices, supports wireless connectivity, touch UI, etc. (e.g., Android, iOS).                                                           |

---

### ✅ **OS Structures**

How the components of an OS are organized.

#### 1. **Monolithic Kernel**

* Entire OS runs in kernel mode.
* All services (I/O, memory, file system) are tightly integrated.
* Example: Linux, UNIX.

#### 2. **Layered Architecture**

* OS is divided into layers, each built on top of lower ones.
* Top: UI Layer → Bottom: Hardware.
* Easier to debug and design.

#### 3. **Microkernel**

* Minimal kernel that handles only core functions (e.g., IPC, basic scheduling).
* Other services run in user space.
* Example: Minix, QNX.

#### 4. **Modular Kernel**

* Combination of Monolithic and Microkernel.
* Supports dynamic loading/unloading of modules.
* Example: Modern Linux kernels.

#### 5. **Hybrid Kernel**

* Uses Microkernel design but incorporates monolithic components for performance.
* Example: Windows NT, macOS.

---

### ✅ **System Calls vs API**

* **System Call:** Interface between application and OS (invokes kernel functions).

  * Example: `fork()`, `exec()`, `read()`
* **API (Application Programming Interface):** High-level interface offered to developers (e.g., POSIX API).

  * System calls are usually **wrapped in APIs** for ease of use.

---

### ✅ **OS Services**

OS provides services to:

* Program Execution
* I/O Operations
* File System Manipulation
* Communication
* Error Detection
* Resource Allocation
* Accounting
* Protection

---

### ✅ **User Mode vs Kernel Mode**

* **User Mode:** Limited access; used by applications.
* **Kernel Mode:** Full system access; used by OS code.
* Mode switching occurs during **system calls or interrupts**.

---

### ✅ **Boot Process**

Steps the OS takes to start a system.

#### 1. **BIOS/UEFI (Firmware)**

* Performs POST (Power-On Self-Test)
* Loads the Bootloader from disk

#### 2. **Bootloader**

* Loads the kernel into memory (e.g., GRUB)

#### 3. **Kernel Initialization**

* Initializes devices, memory management
* Starts system processes (`init` or `systemd`)

---

# 💼 Interview Questions with Answers (Topic-Wise)

---

### 🔹 Q1: What is an operating system?

**A:** An operating system is system software that acts as an intermediary between users and hardware, managing system resources and providing a user interface.

---

### 🔹 Q2: What are the key goals of an OS?

**A:**

* Convenience for users
* Efficient hardware utilization
* Scalability and adaptability for new hardware

---

### 🔹 Q3: What is the difference between kernel mode and user mode?

**A:**

* Kernel mode has unrestricted access to system resources.
* User mode has limited access; user programs run in this mode to protect the system.

---

### 🔹 Q4: Compare Batch OS and Time-Sharing OS.

**A:**

* **Batch OS:** Executes jobs without user interaction; good for large, repetitive jobs.
* **Time-Sharing OS:** Allows multiple users/programs to run concurrently; supports interaction.

---

### 🔹 Q5: What is a system call? Give examples.

**A:**
A system call allows user-level processes to request services from the OS.
**Examples:** `fork()`, `read()`, `write()`, `exec()`, `exit()`.

---

### 🔹 Q6: Explain Monolithic vs Microkernel.

**A:**

* **Monolithic:** All OS services run in kernel space.
* **Microkernel:** Only essential services (like IPC, scheduling) are in kernel space; rest in user space.

---

### 🔹 Q7: What is the difference between API and system call?

**A:**

* **System call** is a low-level request to the OS.
* **API** is a high-level abstraction that internally uses system calls.

---

### 🔹 Q8: Describe the boot process.

**A:**

1. **BIOS/UEFI** performs POST and loads bootloader.
2. **Bootloader** loads the OS kernel into memory.
3. **Kernel** initializes hardware and starts system processes.

---

### 🔹 Q9: What are the services provided by an operating system?

**A:**

* Program execution
* I/O operations
* File system management
* Communication
* Error detection
* Protection and security

---

### 🔹 Q10: What are different types of operating systems?

**A:**

* Batch, Time-Sharing, Distributed, Real-Time, Embedded, Network, and Mobile OS.



# 📘 Operating System – Introduction & Basics

## 1. What is OS?
- Interface between User & Hardware  
- Manages Resources  
- Provides Services  

## 2. Goals & Functions
- Convenience  
- Efficiency  
- Evolvability  

### Core Functions:
- Process Management  
- Memory Management  
- File System Management  
- I/O Management  
- Security & Protection  
- User Interface  

## 3. Types of Operating Systems
- Batch OS  
- Time-Sharing OS  
- Distributed OS  
- Real-Time OS  
  - Hard RTOS  
  - Soft RTOS  
- Embedded OS  
- Network OS  
- Mobile OS  

## 4. OS Structures
- Monolithic Kernel  
- Layered OS  
- Microkernel  
- Modular OS  
- Hybrid Kernel  

## 5. System Calls vs API
- System Call = Interface to Kernel  
- API = Interface to Programmer  
- API wraps System Calls (e.g., POSIX)  

## 6. OS Services
- Program Execution  
- I/O Operations  
- File Manipulation  
- Communication  
- Error Detection  
- Resource Allocation  
- Protection & Security  

## 7. User Mode vs Kernel Mode
- Kernel Mode = Full Access  
- User Mode = Restricted Access  

## 8. Boot Process
- **BIOS/UEFI**
  - Power-On Self-Test (POST)  
  - Loads Bootloader  
- **Bootloader**
  - Loads Kernel into Memory  
- **Kernel Initialization**
  - Initializes Devices  
  - Loads Drivers  
  - Starts `init` / `systemd`  


---
---
# 🧑‍💻 2. Process Management — **Detailed Explanation**

---

## ✅ **Process Concepts**

### 🔹 **What is a Process?**

A **process** is an active instance of a program in execution. It includes the program code and its current activity, represented by:

* Program Counter (PC)
* Stack (function calls)
* Data section (variables)
* Heap (dynamic memory)

---

### 🔹 **Process Control Block (PCB)**

A **PCB** is a data structure maintained by the OS for every process. It stores:

* Process ID (PID)
* Process state
* Program counter
* CPU registers
* Memory management info (page tables)
* Accounting info (CPU time used)
* I/O status info

---

### 🔹 **Process States**

```plaintext
New → Ready → Running → Terminated
             ↑       ↓
          Waiting ← (I/O or event)
```

* **New**: Process is being created.
* **Ready**: Waiting for CPU.
* **Running**: Executing instructions.
* **Waiting**: Waiting for I/O or event.
* **Terminated**: Finished execution.

---

### 🔹 **Context Switching**

When CPU switches from one process to another, the OS must save the **context** (registers, PC, etc.) of the current process and load the context of the next process.

* Overhead: Takes time (no useful work done)
* Enables multitasking

---

## ✅ **Types of Processes**

| Type                           | Description                                                                            |
| ------------------------------ | -------------------------------------------------------------------------------------- |
| **User vs System**             | User processes run in user mode; system processes run in kernel mode.                  |
| **Foreground vs Background**   | Foreground interacts with the user; background does not (e.g., daemons).               |
| **Independent vs Cooperating** | Independent processes don’t share data; cooperating ones do (e.g., via shared memory). |

---

## ✅ **Process Scheduling**

### 🔹 **Scheduling Queues**

* **Job Queue**: All submitted processes.
* **Ready Queue**: Processes ready to run.
* **Device Queue**: Processes waiting for I/O devices.

### 🔹 **Dispatcher**

* Loads selected process from the ready queue to the CPU.
* Performs:

  * Context switch
  * Jump to user mode
  * Restart program counter

### 🔹 **Preemptive vs Non-Preemptive Scheduling**

| Type               | Description                                    |
| ------------------ | ---------------------------------------------- |
| **Preemptive**     | CPU can be taken away (e.g., Round Robin).     |
| **Non-Preemptive** | CPU runs until the process finishes or blocks. |

---

## 🔁 **Scheduling Algorithms**

### 1. **FCFS (First Come First Serve)**

* Jobs are executed in the order they arrive.
* Non-preemptive
* Simple but may cause **convoy effect** (long process delays short ones).

### 2. **SJF (Shortest Job First)**

* Executes the shortest job first.
* Non-preemptive
* Optimal in terms of **average waiting time**, but needs to know burst time in advance.

### 3. **SRTF (Shortest Remaining Time First)**

* Preemptive version of SJF.
* If a new process arrives with a shorter burst time, it preempts the current one.

### 4. **Priority Scheduling**

* Each process is assigned a priority; highest priority runs first.
* Can be **preemptive or non-preemptive**.
* Problem: **Starvation** (low-priority processes may never run).
* Solution: **Aging** (gradually increasing priority over time).

### 5. **Round Robin (RR)**

* Each process gets a fixed **time quantum**.
* After time expires, it's moved to the back of the queue.
* Fair and preemptive.
* Performance depends on the time quantum.

### 6. **Multilevel Queue Scheduling**

* Multiple queues (foreground, background) with different scheduling algorithms.
* No movement between queues.

### 7. **Multilevel Feedback Queue**

* Like Multilevel Queue, but allows **movement between queues** based on behavior (e.g., CPU-bound vs I/O-bound).
* Adaptive and complex.

### 8. **Lottery Scheduling**

* Each process gets a number of "lottery tickets."
* Scheduler picks a ticket at random.
* Fair, probabilistic, used in some modern systems.

---

# 💼 Interview Questions & Answers

---

### 🔹 Q1: What is a process?

**A:** A process is a program in execution, containing its code, data, stack, and state. It’s managed by the OS through a Process Control Block (PCB).

---

### 🔹 Q2: What is the difference between a program and a process?

**A:** A program is a passive set of instructions; a process is an active execution of those instructions.

---

### 🔹 Q3: What does a Process Control Block (PCB) contain?

**A:** PID, process state, CPU registers, program counter, memory info, I/O status, and accounting info.

---

### 🔹 Q4: What is context switching?

**A:** Context switching is saving the state of a running process and loading the state of another, enabling multitasking.

---

### 🔹 Q5: Difference between preemptive and non-preemptive scheduling?

**A:** Preemptive allows the OS to interrupt a running process; non-preemptive lets the process finish or block first.

---

### 🔹 Q6: Which scheduling algorithm is best for minimum average waiting time?

**A:** SJF (Shortest Job First), assuming burst times are known.

---

### 🔹 Q7: What is starvation in scheduling?

**A:** Starvation happens when low-priority processes are indefinitely delayed. Aging can solve this.

---

### 🔹 Q8: Explain the Round Robin algorithm.

**A:** Processes are given a fixed time slice in rotation. Fair and preemptive, ideal for time-sharing systems.

---

### 🔹 Q9: What is the difference between Multilevel Queue and Multilevel Feedback Queue?

**A:**

* **Multilevel Queue**: Fixed queues, no movement.
* **Feedback Queue**: Processes can move between queues.

---

### 🔹 Q10: How is lottery scheduling different from traditional scheduling?

**A:** Lottery scheduling assigns tickets to processes and picks randomly. It’s probabilistic and allows fair CPU sharing.

---

# 🧵 **3. Thread Management — Detailed Explanation**

---

## ✅ **Threads vs Processes**

| Feature       | Process                                                            | Thread                                                                                           |
| ------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Definition    | A process is an independent unit of execution with its own memory. | A thread is a lightweight sub-process that shares memory with other threads in the same process. |
| Overhead      | High (requires more resources)                                     | Low (lighter, faster to switch)                                                                  |
| Isolation     | Fully isolated from others                                         | Shares memory space                                                                              |
| Communication | Via IPC (Inter-Process Communication)                              | Easier through shared memory                                                                     |
| Crashes       | One process crash doesn't affect others                            | One thread crash can affect entire process                                                       |

---

## ✅ **User-Level vs Kernel-Level Threads**

| Feature           | User-Level Threads (ULT)      | Kernel-Level Threads (KLT)               |
| ----------------- | ----------------------------- | ---------------------------------------- |
| Managed by        | User-level thread library     | Kernel                                   |
| Context Switching | Fast (no mode switching)      | Slower (requires kernel involvement)     |
| Kernel Visibility | Not visible to OS             | Visible to OS                            |
| Blocking Behavior | One blocked thread blocks all | One blocked thread doesn't affect others |
| Examples          | POSIX pthreads, Java threads  | Windows NT threads, Linux kernel threads |

---

## ✅ **Thread Libraries**

Libraries abstract thread creation and management for developers.

1. **POSIX Threads (pthreads)**:

   * Portable, Unix-based
   * `pthread_create()`, `pthread_join()`, etc.

2. **Windows Threads**:

   * Windows API for thread creation: `CreateThread()`, `WaitForSingleObject()`

3. **Java Threads**:

   * High-level abstraction
   * Created via extending `Thread` class or implementing `Runnable` interface

---

## ✅ **Multithreading Models**

### 1. **Many-to-One**

* Many user threads map to **one kernel thread**
* Poor concurrency (one thread blocks all)
* Simple to manage
* Example: Green Threads in older Java

### 2. **One-to-One**

* Each user thread maps to **one kernel thread**
* Better concurrency
* Expensive due to more kernel resources
* Example: Windows OS, Linux (pthreads)

### 3. **Many-to-Many**

* Many user threads mapped to **many kernel threads**
* Best of both worlds
* Efficient scheduling and better concurrency
* Example: Solaris, Windows with fiber support

---

## ✅ **Thread Pooling**

* A collection of pre-initialized threads ready to be assigned tasks.
* Improves performance by reusing threads instead of creating/destroying them.
* Useful in server applications (e.g., web servers, database servers).

**Advantages:**

* Reduces latency
* Saves system resources
* Prevents overload

---

## ✅ **Thread Synchronization**

Multiple threads may access shared data → **Race Conditions** occur.

### Synchronization Mechanisms:

* **Mutexes (Mutual Exclusion)**:

  * Ensures only one thread accesses critical section at a time
* **Semaphores**:

  * Generalized locking, can count resources
* **Monitors**:

  * High-level abstraction (automatic locking)
* **Condition Variables**:

  * Used for signaling between threads

---

## ✅ **Thread Scheduling**

* Managed by kernel (KLT) or thread library (ULT)
* Can follow policies like:

  * Round Robin
  * Priority-based
  * FIFO (First-In-First-Out)
* OS schedules threads based on fairness, priority, and real-time constraints.

---

# 💼 Interview Questions with Answers

---

### 🔹 Q1: What is the difference between a thread and a process?

**A:** A process is an independent execution unit with its own memory, while a thread is a lightweight execution unit that shares memory with other threads in the same process.

---

### 🔹 Q2: What is a thread library?

**A:** A thread library provides APIs to create, manage, and synchronize threads. Examples include POSIX threads (`pthreads`), Windows Threads, and Java Threads.

---

### 🔹 Q3: Explain the difference between user-level and kernel-level threads.

**A:**

* **User-Level Threads (ULT):** Managed in user space, fast context switching, not visible to the kernel.
* **Kernel-Level Threads (KLT):** Managed by OS kernel, better concurrency, more overhead.

---

### 🔹 Q4: What are the different multithreading models?

**A:**

* **Many-to-One**: Many user threads → 1 kernel thread (low concurrency)
* **One-to-One**: Each user thread → 1 kernel thread (good concurrency)
* **Many-to-Many**: Many user threads → Many kernel threads (efficient & flexible)

---

### 🔹 Q5: What is thread pooling?

**A:** Thread pooling is a technique where a fixed set of reusable threads is maintained to execute tasks, reducing overhead of thread creation and destruction.

---

### 🔹 Q6: What is a race condition?

**A:** A race condition occurs when multiple threads access shared data simultaneously and the result depends on the timing of their execution.

---

### 🔹 Q7: How do you prevent race conditions?

**A:** Use synchronization tools like Mutexes, Semaphores, and Monitors to protect critical sections.

---

### 🔹 Q8: What is the role of a monitor?

**A:** A monitor is a high-level synchronization construct that encapsulates shared variables, the procedures that operate on them, and synchronization between threads.

---

### 🔹 Q9: Which model offers the best scalability: Many-to-One, One-to-One, or Many-to-Many?

**A:** **Many-to-Many** offers the best scalability and performance because it allows flexibility in mapping threads and avoids resource exhaustion.

---

### 🔹 Q10: What happens if one thread in a user-level thread model blocks?

**A:** If one thread blocks in a **Many-to-One model**, all threads block because the kernel sees them as a single thread.

---

**Thread Management** topics using `POSIX Threads (pthreads)` — the standard for multithreading in C on UNIX/Linux systems.

---

# ✅ **3. Thread Management in C with Code Examples (Using `pthreads`)**

---

## 1️⃣ **Threads vs Processes**

(We demonstrate thread creation and compare with processes.)

### 🧵 **Thread Example (Using `pthread_create`)**

```c
#include <pthread.h>
#include <stdio.h>

void* myThreadFunc(void* arg) {
    printf("Hello from thread!\n");
    return NULL;
}

int main() {
    pthread_t tid;
    pthread_create(&tid, NULL, myThreadFunc, NULL);
    pthread_join(tid, NULL);
    printf("Thread has finished execution.\n");
    return 0;
}
```

---

### 🔄 **Process Example (Using `fork`)**

```c
#include <stdio.h>
#include <unistd.h>

int main() {
    pid_t pid = fork();
    if (pid == 0)
        printf("Child Process\n");
    else
        printf("Parent Process\n");
    return 0;
}
```

---

## 2️⃣ **User-Level vs Kernel-Level Threads**

In Linux with `pthreads`, you create kernel-level threads. We cannot fully show ULTs in C alone, but you can **observe the difference via thread IDs**:

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

void* threadFunc(void* arg) {
    printf("Inside thread. TID: %lu\n", pthread_self());
    return NULL;
}

int main() {
    pthread_t tid;
    pthread_create(&tid, NULL, threadFunc, NULL);
    pthread_join(tid, NULL);
    printf("Main thread finished.\n");
    return 0;
}
```

---

## 3️⃣ **Thread Libraries: POSIX (pthreads)**

* Already demonstrated above using:

  * `pthread_create`
  * `pthread_join`

Other useful functions:

```c
pthread_exit();    // Exit thread
pthread_self();    // Get thread ID
pthread_equal();   // Compare thread IDs
```

---

## 4️⃣ **Multithreading Models (Simulated)**

In C with `pthreads`, only **one-to-one** is directly supported (each pthread maps to a kernel thread). Simulation of models:

### ✅ Many-to-One: (Simulated with one kernel thread and manual switch)

Not directly implementable in `pthreads`.

### ✅ One-to-One: Default `pthread_create`

### ✅ Many-to-Many: Requires custom threading libraries or OS support (e.g., GNU Portable Threads), not native to C with pthreads.

---

## 5️⃣ **Thread Pooling Example (Simple Simulation)**

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define THREAD_COUNT 3

void* task(void* arg) {
    int* id = (int*)arg;
    printf("Thread %d is doing its work.\n", *id);
    free(id);
    return NULL;
}

int main() {
    pthread_t threads[THREAD_COUNT];

    for (int i = 0; i < THREAD_COUNT; i++) {
        int* id = malloc(sizeof(int));
        *id = i + 1;
        pthread_create(&threads[i], NULL, task, id);
    }

    for (int i = 0; i < THREAD_COUNT; i++)
        pthread_join(threads[i], NULL);

    return 0;
}
```

---

## 6️⃣ **Thread Synchronization**

### 🔐 **Using Mutex**

```c
#include <pthread.h>
#include <stdio.h>

int counter = 0;
pthread_mutex_t lock;

void* increment(void* arg) {
    for (int i = 0; i < 100000; i++) {
        pthread_mutex_lock(&lock);
        counter++;
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;
    pthread_mutex_init(&lock, NULL);

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    pthread_mutex_destroy(&lock);
    printf("Final counter value: %d\n", counter);
    return 0;
}
```

---

### ⚖️ **Using Semaphore**

```c
#include <pthread.h>
#include <stdio.h>
#include <semaphore.h>

int counter = 0;
sem_t sem;

void* increment(void* arg) {
    for (int i = 0; i < 100000; i++) {
        sem_wait(&sem);
        counter++;
        sem_post(&sem);
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;
    sem_init(&sem, 0, 1);

    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    sem_destroy(&sem);
    printf("Final counter value: %d\n", counter);
    return 0;
}
```

---

## 7️⃣ **Thread Scheduling (Set Priority — Limited Control in Linux)**

```c
#include <pthread.h>
#include <stdio.h>
#include <sched.h>

void* threadFunc(void* arg) {
    printf("Running thread...\n");
    return NULL;
}

int main() {
    pthread_t tid;
    pthread_attr_t attr;
    struct sched_param param;

    pthread_attr_init(&attr);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO); // FIFO policy
    param.sched_priority = 10;
    pthread_attr_setschedparam(&attr, &param);

    pthread_create(&tid, &attr, threadFunc, NULL);
    pthread_join(tid, NULL);
    return 0;
}
```

⚠️ **Note**: You may need **root privileges** to change scheduling policies or priorities.

---

## 🧪 Compile All C Codes Like:

```bash
gcc filename.c -o output -lpthread
./output
```

---





