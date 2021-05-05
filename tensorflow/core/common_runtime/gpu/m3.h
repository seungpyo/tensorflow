#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <mutex>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <atomic>
#include <semaphore.h>
#include "cuda.h"
#include "cuutils.h"

namespace M3 {

typedef uintptr_t shareable_handle_t;

enum Cmd {
	M3_INVALID_CMD,
	M3_MEM_CREATE,
	M3_HALT
};

#define MAX_MEMID_LEN 1024

typedef struct Request {
	Cmd cmd;
	size_t num_bytes;
	size_t alignment;
	char mem_id[MAX_MEMID_LEN];
	Request() = default;
	Request(Cmd _cmd, size_t _num_bytes, size_t _alignment, char * _mem_id) :
		cmd(_cmd), num_bytes(_num_bytes), alignment(_alignment) {
		strncpy(mem_id, _mem_id, MAX_MEMID_LEN);
	}
	Request(Cmd _cmd, size_t _num_bytes, size_t _alignment, const char * _mem_id) :
		cmd(_cmd), num_bytes(_num_bytes), alignment(_alignment) {
		strncpy(mem_id, _mem_id, MAX_MEMID_LEN);
	}

} Request;

enum Status {
	M3_NYI,
	M3_ACK,
	M3_INVALID_ARGUMENT,
	M3_OUT_OF_MEMORY,
	M3_SYSCALL_FAILURE,
	M3_SYSCALL_SENDTO_FAILURE,
	M3_SYSCALL_RECVFROM_FAILURE,
	M3_SYSCALL_RECVHANDLE_FAILURE
};

typedef struct Response {
	Status status;
	shareable_handle_t sh_handle;
	size_t recv_size;
	Response() = default;
	Response(Status _status, shareable_handle_t _sh_handle, size_t _recv_size) :
		status(_status), sh_handle(_sh_handle), recv_size(_recv_size) {}
} Response;

Status RemoteMemCreate(size_t num_bytes, size_t alignment, char *mem_id, Response &res);

class Server {
	public:
		Server();
		void Run();

	private:
		CUmemAllocationProp initProp() {
			CUmemAllocationProp prop = {};
			prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
			prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
			prop.location.id = 0;
			prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
			return prop;
		}

		int sock_fd_;
		Request req_;
		Response res_; 
		struct sockaddr_un cli_addr_;
		socklen_t cli_addr_len_;
		
		size_t granularity_;
		CUmemAllocationProp prop_;

    struct pairHash {
      template <class T1, class T2>
      std::size_t operator () (const std::pair<T1,T2> &p) const {
          auto h1 = std::hash<T1>{}(p.first);
          auto h2 = std::hash<T2>{}(p.second);
          return h1 ^ h2;  
      }
    };

		std::unordered_map<CUmemGenericAllocationHandle, std::vector<shareable_handle_t>> phys2shHandle_;
		std::unordered_map<std::pair<std::string, size_t>, CUmemGenericAllocationHandle, pairHash> memId2phys_;
};

static struct sockaddr_un srv_addr = {
	AF_UNIX,
	"/tmp/M3_IPC_EndPoint"
};
static socklen_t srv_addr_len = sizeof(srv_addr);

// panic() ; You know what it does.
void panic(const char * msg);

// Helper functions for IPC features.

// Semaphore locking / unlocking helper functions.

int ipcLockGeneric(int initialValue);
int ipcLock(void);
int ipcUnlock(void);

// ipcLockPrivileged() is used only by server, 
// just to protect server initialization process done in MemMapManager::MemMapManager().
int ipcLockPrivileged(void);

int ipcOpenAndBindSocket(struct sockaddr_un * local_addr);

// ipcSendShareableHandle() sends multiple shareable handles (UNIX file descriptors) using sendmsg().
// this function is used by RequestAllocate().
int ipcSendShareableHandle(int sock_fd, struct sockaddr_un * client_addr, shareable_handle_t shHandle);

// ipcRecvShareableHandle() receives multiple shareable handles (UNIX file descriptors) using recvmsg()
// this function is used by RequestAllocate().
int ipcRecvShareableHandle(int sock_fd, shareable_handle_t *shHandle);

} /* namespace M3 */
