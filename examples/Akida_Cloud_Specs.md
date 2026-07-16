# 📄⚙️ Akida Pico FPGA — Technical Specifications

These specs describe the **Akida Pico FPGA cloud platform** — a single-NP Akida Pico IP implemented on a Xilinx FPGA, hosted as a remote workstation accessible via Jupyter Lab.

---

### **Akida Pico IP Configuration**

| **Parameter** | **Specification** |
|---------------|-------------------|
| **IP Version** | Akida **Pico** (`IpVersion.pico`) |
| **Device Version** | `BC.B1.001.000` |
| **Vendor ID / Product ID** | 188 / 177 |
| **Nodes** | 1 |
| **Neural Processors (NPs)** | 1 (`Type.TNP_R`) |
| **Reference Clock** | 50 MHz nominal (configurable 25–100 MHz in the demos) |
| **Akida Runtime** | `akida` 2.19.1 |
| **Documentation** | [Akida Documentation](https://doc.brainchipinc.com/index.html#overview) |

---

### **FPGA Hardware**

| **Component** | **Specification** |
|---------------|-------------------|
| **FPGA Vendor** | Xilinx |
| **PCIe Device ID** | `10ee:4b28` (subsystem `10ee:4340`) |
| **Form Factor** | PCIe card |
| **Host Driver** | `xdma` (kernel modules: `xdma`, `xdma_akida`, `xdma_aethercore`) |
| **BARs** | 4 MB main region + 64 KB control region |

---

### **Host System Specifications**

| **Component** | **Specification** |
|---------------|-------------------|
| **Operating System** | Ubuntu 22.04.5 LTS x86_64 |
| **Kernel Version** | 6.8.0-60-generic |
| **Processor** | 11th Gen Intel Core i7-11700B @ 3.20 GHz (boost 4.90 GHz) |
| **CPU Cores / Threads** | 8 cores / 16 threads (1 socket) |
| **Memory** | 62 GB |

---

### **Development Environment**

| **Tool** | **Version/Type** |
|----------|------------------|
| **Interface** | Jupyter Lab |
| **Python Support** | Full Python ecosystem (conda environments) |
| **Terminal Access** | Bash shell via web interface |
| **File Transfer** | Web-based drag & drop |
| **Code Examples** | Pre-loaded demonstration notebooks (KWS, etc.) |
| **System Binaries** | `ffmpeg` / `ffprobe` (required by `tensorflow_datasets` for first-run dataset prep) |

---


*For detailed technical documentation and API references, visit the [Akida Documentation Portal](https://doc.brainchipinc.com/index.html#overview)*
