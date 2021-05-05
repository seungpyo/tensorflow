#include "m3.h"

#include "tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h"

#include "absl/strings/str_format.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/stream_executor/lib/status.h"


namespace M3 {

extern struct sockaddr_un srv_addr;
extern socklen_t srv_addr_len;

Status RemoteMemCreate(size_t num_bytes, size_t alignment, char *mem_id, Response& res) {
	Request req(M3_MEM_CREATE, num_bytes, alignment, mem_id);

	struct sockaddr_un client_addr;
	client_addr.sun_family = AF_UNIX;
	sprintf(client_addr.sun_path, "/tmp/m3_ipc_ep_%d", getpid());
	unlink(client_addr.sun_path);		

	int sock_fd = ipcOpenAndBindSocket(&client_addr);
	if (sock_fd < 0) {
    LOG(ERROR) << "Failed to open and bind socket to " << client_addr.sun_path;
		res.status = M3_SYSCALL_FAILURE;
	}
	if (sendto(sock_fd, (const void *)&req, sizeof(req), 0, (struct sockaddr *)&srv_addr, srv_addr_len) < 0) {
		LOG(ERROR) << "Failed to send to endpoint " << srv_addr.sun_path;
		res.status = M3_SYSCALL_SENDTO_FAILURE;
	}	
	if (recvfrom(sock_fd, (void *)&res, sizeof(res), 0, (struct sockaddr *)&srv_addr, &srv_addr_len) < 0) {
		res.status =  M3_SYSCALL_RECVFROM_FAILURE;
	}
	if (res.status != M3_ACK) {
     LOG(ERROR) << "Unknown error in M3 Server, returning code " << res.status;
	}
	
	if (ipcRecvShareableHandle(sock_fd, &res.sh_handle) < 0) {
    LOG(ERROR) << "Failed to receive shareable handle (fd)";
		res.status = M3_SYSCALL_RECVHANDLE_FAILURE;
	}

	return res.status;
}

Server::Server() {
	unlink(srv_addr.sun_path);
	sock_fd_ = ipcOpenAndBindSocket(&srv_addr);
	cli_addr_len_ = sizeof(cli_addr_);

	CUUTIL_ERRCHK( cuInit(0) );	

	CUmemAllocationProp prop = initProp();
	CUUTIL_ERRCHK( cuMemGetAllocationGranularity(&granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM)  );
  std::cout << "M3 Server granularity = " << granularity_ << std::endl;
	prop = initProp();
	CUmemGenericAllocationHandle ah;
	CUUTIL_ERRCHK( cuMemCreate(&ah, 0x200000, &prop, 0) );

}

void Server::Run() {
	bool run = true;
	while(run) {
		if (recvfrom(sock_fd_, (void *)&req_, sizeof(req_), 0, (struct sockaddr *)&cli_addr_, &cli_addr_len_) < 0) {
			res_.status = M3_SYSCALL_FAILURE;
		} else {
			res_.status = M3_ACK;
		}

		std::string memIdStr;
		size_t roundedBytes;
		CUmemGenericAllocationHandle allocHandle;
    std::pair<std::string, size_t> queryPair;
		switch(req_.cmd) {
			case M3_MEM_CREATE:
				memIdStr = req_.mem_id;
				if(req_.num_bytes < 1) {
					res_.status = M3_INVALID_ARGUMENT;
					break;
				}
				roundedBytes = ((req_.num_bytes + granularity_ - 1) / granularity_) * granularity_;
				res_.recv_size = roundedBytes;

        queryPair = std::make_pair(memIdStr, roundedBytes);
				if (memId2phys_.count(queryPair) > 0) {
					allocHandle = memId2phys_[queryPair];
				} else {
					CUmemAllocationProp prop = initProp();
					CUUTIL_ERRCHK( cuMemCreate(&allocHandle, roundedBytes, &prop, 0) );
					memId2phys_[queryPair] = allocHandle;
				}
				CUUTIL_ERRCHK( cuMemExportToShareableHandle((void *)&res_.sh_handle, allocHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) );
				phys2shHandle_[allocHandle].push_back(res_.sh_handle);
				break;
			case M3_HALT:
				run = false;
				break;
			default:
				res_.status = M3_NYI;
		}

		if (sendto(sock_fd_, (const void *)&res_, sizeof(res_), 0, (struct sockaddr *)&cli_addr_, cli_addr_len_) < 0) {
			res_.status = M3_SYSCALL_FAILURE;
		}
		if (req_.cmd == M3_MEM_CREATE) {
			if(ipcSendShareableHandle(sock_fd_, &cli_addr_, res_.sh_handle) < 0) {
				panic("Sending shareable handle");
			}
		}
	}
}


} /* namespace M3 */

