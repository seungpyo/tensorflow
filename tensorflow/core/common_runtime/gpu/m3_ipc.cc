#include "m3.h"

namespace M3 {

const char ipcEndpointName[] = "M3_IPC_ENDPOINT";
const char ipcLockName[] = "M3_IPC_LOCK";

void panic(const char * msg) {

    perror(msg);
    exit(EXIT_FAILURE);

}

int ipcSendShareableHandle(int sock_fd, struct sockaddr_un * client_addr, shareable_handle_t shHandle) {

    struct msghdr msg;
    struct iovec iov[1];

    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    ssize_t readResult;
    socklen_t len = sizeof(*client_addr);

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &shHandle, sizeof(shHandle));

    msg.msg_name = (void *)client_addr;
    msg.msg_namelen = sizeof(struct sockaddr_un);

    iov[0].iov_base = (void *)"";
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t sendResult = sendmsg(sock_fd, &msg, 0);
    if (sendResult <= 0) {
        perror("IPC failure: Sending data over socket failed");
        return -1;
    }
    return 0;

}

int ipcRecvShareableHandle(int sock_fd, shareable_handle_t *shHandle) {
    struct msghdr msg = {0};
    struct iovec iov[1];
    struct cmsghdr cm;

    // Union to guarantee alignment requirements for control array
    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    ssize_t n;
    int receivedfd;
    char dummy_buffer[1];
    ssize_t sendResult;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    if ((n = recvmsg(sock_fd, &msg, 0)) <= 0) {
        perror("IPC failure: Receiving data over socket failed");
        return -1;
    }

    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
        (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
    if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        return -1;
    }

    memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
    *(int *)shHandle = receivedfd;
    } else {
    return -1;
    }

    return 0;
}


int ipcOpenAndBindSocket(struct sockaddr_un * local_addr) {
    int sock_fd;
    if((sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1) {
        panic("MemMapManager: Failed to open server socket");
    }

    if (bind(sock_fd, (struct sockaddr *)local_addr, SUN_LEN(local_addr)) < 0) {
        printf("[PID = %d] Failed to bind IPC socket to local_addr = %s\n", getpid(), local_addr->sun_path);
        panic("");
    }

    return sock_fd;
}


int ipcLockGeneric(int initialValue) {
    sem_t * sem = sem_open(ipcLockName, O_CREAT, S_IRUSR|S_IWUSR, initialValue);
    if (sem == SEM_FAILED) {
        std::cout << "MemMapManager: Failed to open semaphore" << ipcLockName << std::endl;
        return -1;
    }
    sem_wait(sem);
    sem_close(sem); 
    return 0;
}

int ipcLock() {
    return ipcLockGeneric(0);
}

int ipcLockPrivileged() {
    return ipcLockGeneric(1);
}


int ipcUnlock(void) {
    sem_t * sem = sem_open(ipcLockName, O_CREAT, S_IRUSR|S_IWUSR, 0);
    if (sem == SEM_FAILED) {
        std::cout << "MemMapManager: Failed to open semaphore" << ipcLockName << std::endl;
        return -1;
    }
    sem_post(sem);
    sem_close(sem); 
    return 0;
}

} /* namespace M3 */
