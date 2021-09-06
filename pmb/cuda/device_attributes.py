#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package pmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
class CudaDeviceAttributes:
    """Provide an easy access to all cuda device attributes."""

    def __init__(self, attributes, properties):
        """Initialize a CudaDeviceAttributes from a cupy device attribute dictionnary."""
        self._attributes = attributes
        self._properties = properties

    @property
    def name(self):
        """Returns device name."""
        return self._properties["name"].decode("utf-8")

    @property
    def uuid(self):
        """Returns device uuid (16B unique identifier)."""
        return self._properties["uuid"]

    @property
    def luid(self):
        """Returns device LUID (8B locally unique identifier). Value is undefined on TCC and non-Windows platforms."""
        return self._properties["luid"]

    @property
    def luid_node_mask(self):
        """Returns LUID device node mask. Value is undefined on TCC and non-Windows platforms."""
        return self._properties["luidDeviceNodeMask"]

    @property
    def persisting_l2_cache_max_size(self):
        """Device's maximum L2 persisting lines capacity setting in bytes."""
        return self._properties["persistingL2CacheMaxSize"]

    @property
    def max_blocks_per_multi_processor(self):
        """Maximum number of resident blocks per multiprocessor."""
        return self._properties["maxBlocksPerMultiProcessor"]

    @property
    def access_policy_max_window_size(self):
        """Returns the maximum value of cudaAccessPolicyWindow.num_bytes."""
        return self._properties["accessPolicyMaxWindowSize"]

    @property
    def reserved_shared_memory_per_block(self):
        """Shared memory reserved by CUDA driver per block in bytes."""
        return self._properties["reservedSharedMemPerBlock"]

    @property
    def total_global_memory(self):
        """Returns total global memory available on device in bytes."""
        return self._properties["totalGlobalMem"]

    @property
    def compute_major(self):
        """Returns major compute capability."""
        return self._properties["major"]

    @property
    def compute_minor(self):
        """Returns major compute capability."""
        return self._properties["minor"]

    @property
    def async_engine_count(self):
        """Returns the number of async engine count of the device."""
        return self._attributes["AsyncEngineCount"]

    @property
    def can_flush_remote_writes(self):
        """If set, device can flush remote writes."""
        return bool(self._attributes["CanFlushRemoteWrites"])

    @property
    def can_map_host_memory(self):
        """If set, device can use pinned memory."""
        return bool(self._attributes["CanMapHostMemory"])

    @property
    def can_use_host_pointer_for_registered_mem(self):
        """If set, device can use a host pointer as registered memory."""
        return bool(self._attributes["CanUseHostPointerForRegisteredMem"])

    @property
    def clock_rate(self):
        """Device clock frequency in kilohertz."""
        return self._attributes["ClockRate"]

    @property
    def compute_mode(self):
        """Returns device compute mode."""
        return self._attributes["ComputeMode"]

    @property
    def compute_preemption_supported(self):
        """If set, device supports compute pre-emption."""
        return bool(self._attributes["ComputePreemptionSupported"])

    @property
    def concurrent_kernels(self):
        """Max number of concurrent kernels."""
        return self._attributes["ConcurrentKernels"]

    @property
    def concurrent_managed_access(self):
        """Max number of concurrent managed memory accesses."""
        return self._attributes["ConcurrentManagedAccess"]

    @property
    def cooperative_launch(self):
        """If set, device supports launching cooperative kernels."""
        return bool(self._attributes["CooperativeLaunch"])

    @property
    def cooperative_multi_device_launch(self):
        """If set, device can participate in cooperative multi-device kernels."""
        return bool(self._attributes["CooperativeMultiDeviceLaunch"])

    @property
    def direct_managed_mem_access_from_host(self):
        """If set, host can directly access managed memory."""
        return bool(self._attributes["DirectManagedMemAccessFromHost"])

    @property
    def ecc_enabled(self):
        """If set, device supports ECC and has error correction enabled."""
        return bool(self._attributes["EccEnabled"])

    @property
    def global_l1_cache_supported(self):
        """If set, device supports caching globals in L1 cache."""
        return bool(self._attributes["GlobalL1CacheSupported"])

    @property
    def global_memory_bus_width(self):
        """Returns device global memory bus width in bits."""
        return self._attributes["GlobalMemoryBusWidth"]

    @property
    def gpu_overlap(self):
        """If set, the device can concurrently copy memory and execute a kernel."""
        return bool(self._attributes["GpuOverlap"])

    @property
    def host_native_atomic_supported(self):
        """If set, atomic operations over the link are supported."""
        return bool(self._attributes["HostNativeAtomicSupported"])

    @property
    def host_register_supported(self):
        """If set, device supports host memory registration via cudaHostRegister."""
        return bool(self._attributes["HostRegisterSupported"])

    @property
    def integrated(self):
        """If set, device memory is integrated with host memory."""
        return bool(self._attributes["Integrated"])

    @property
    def is_multi_gpu_board(self):
        """If set, device is on a multi-GPU board."""
        return bool(self._attributes["IsMultiGpuBoard"])

    @property
    def kernel_exec_timeout(self):
        """Specifies whether there is a run time limit on kernels."""
        return bool(self._attributes["KernelExecTimeout"])

    @property
    def l2_cache_size(self):
        """Size of device L2 cache in bytes."""
        return self._attributes["L2CacheSize"]

    @property
    def local_l1_cache_supported(self):
        """If set, device supports caching locals in L1 cache."""
        return bool(self._attributes["LocalL1CacheSupported"])

    @property
    def managed_memory(self):
        """If set, device can allocate managed memory on this system."""
        return bool(self._attributes["ManagedMemory"])

    @property
    def max_block_dim_x(self):
        """Returns device maximum block dimension in x."""
        return self._attributes["MaxBlockDimX"]

    @property
    def max_block_dim_y(self):
        """Returns device maximum block dimension in y."""
        return self._attributes["MaxBlockDimY"]

    @property
    def max_block_dim_z(self):
        """Returns device maximum block dimension in z."""
        return self._attributes["MaxBlockDimZ"]

    @property
    def max_grid_dim_x(self):
        """Returns device maximum grid dimension in x."""
        return self._attributes["MaxGridDimX"]

    @property
    def max_grid_dim_y(self):
        """Returns device maximum grid dimension in y."""
        return self._attributes["MaxGridDimY"]

    @property
    def max_grid_dim_z(self):
        """Returns device maximum grid dimension in z."""
        return self._attributes["MaxGridDimZ"]

    @property
    def max_pitch(self):
        """Maximum pitch in bytes allowed by memory copies."""
        return self._attributes["MaxPitch"]

    @property
    def max_registers_per_block(self):
        """Maximum number of 32-bit registers available per block."""
        return self._attributes["MaxRegistersPerBlock"]

    @property
    def max_registers_per_multiprocessor(self):
        """Maximum number of 32-bit registers available per multiprocessor."""
        return self._attributes["MaxRegistersPerMultiprocessor"]

    @property
    def max_shared_memory_per_block(self):
        """Maximum shared memory available per block in bytes."""
        return self._attributes["MaxSharedMemoryPerBlock"]

    @property
    def max_shared_memory_per_block_optin(self):
        """Maximum optin shared memory per block."""
        return self._attributes["MaxSharedMemoryPerBlockOptin"]

    @property
    def max_shared_memory_per_multiprocessor(self):
        """Maximum shared memory available per multiprocessor in bytes."""
        return self._attributes["MaxSharedMemoryPerMultiprocessor"]

    @property
    def max_surface1D_layered_layers(self):
        """Maximum layers in a 1D layered surface."""
        return self._attributes["MaxSurface1DLayeredLayers"]

    @property
    def max_surface1D_layered_width(self):
        """Maximum 1D layered surface width."""
        return self._attributes["MaxSurface1DLayeredWidth"]

    @property
    def max_surface1D_width(self):
        """Maximum 1D surface width."""
        return self._attributes["MaxSurface1DWidth"]

    @property
    def max_surface2D_height(self):
        """Maximum 2D surface height."""
        return self._attributes["MaxSurface2DHeight"]

    @property
    def max_surface2D_layered_height(self):
        """Maximum 2D layered surface height."""
        return self._attributes["MaxSurface2DLayeredHeight"]

    @property
    def max_surface2D_layered_layers(self):
        """Maximum layers in a 2D layered surface."""
        return self._attributes["MaxSurface2DLayeredLayers"]

    @property
    def max_surface2D_layered_width(self):
        """Maximum 2D layered surface width."""
        return self._attributes["MaxSurface2DLayeredWidth"]

    @property
    def max_surface2D_width(self):
        """Maximum 2D surface width."""
        return self._attributes["MaxSurface2DWidth"]

    @property
    def max_surface3D_depth(self):
        """Maximum 3D surface depth."""
        return self._attributes["MaxSurface3DDepth"]

    @property
    def max_surface3D_height(self):
        """Maximum 3D surface height."""
        return self._attributes["MaxSurface3DHeight"]

    @property
    def max_surface3D_width(self):
        """Maximum 3D surface width."""
        return self._attributes["MaxSurface3DWidth"]

    @property
    def max_surface_cubemap_layered_layers(self):
        """Maximum layers in a cubemap layered surface."""
        return self._attributes["MaxSurfaceCubemapLayeredLayers"]

    @property
    def max_surface_cubemap_layered_width(self):
        """Maximum cubemap layered surface width."""
        return self._attributes["MaxSurfaceCubemapLayeredWidth"]

    @property
    def max_surface_cubemap_width(self):
        """Maximum cubemap surface width."""
        return self._attributes["MaxSurfaceCubemapWidth"]

    @property
    def max_texture1D_layered_layers(self):
        """Maximum layers in a 1D layered texture."""
        return self._attributes["MaxTexture1DLayeredLayers"]

    @property
    def max_texture1D_layered_width(self):
        """Maximum 1D layered texture width."""
        return self._attributes["MaxTexture1DLayeredWidth"]

    @property
    def max_texture1D_linear_width(self):
        """
        Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth()
        or cuDeviceGetTexture1DLinearMaxWidth() instead which returns the
        maximum number of elements allocatable in a 1D linear texture for a
        given element size.
        """
        return self._attributes["MaxTexture1DLinearWidth"]

    @property
    def max_texture1D_mipmapped_width(self):
        """Maximum mipmapped 1D texture width."""
        return self._attributes["MaxTexture1DMipmappedWidth"]

    @property
    def max_texture1D_width(self):
        """Maximum 1D texture width."""
        return self._attributes["MaxTexture1DWidth"]

    @property
    def max_texture2D_gather_height(self):
        """Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set."""
        return self._attributes["MaxTexture2DGatherHeight"]

    @property
    def max_texture2D_gather_width(self):
        """Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set."""
        return self._attributes["MaxTexture2DGatherWidth"]

    @property
    def max_texture2D_height(self):
        """Maximum 2D texture height."""
        return self._attributes["MaxTexture2DHeight"]

    @property
    def max_texture2D_layered_height(self):
        """Maximum 2D layered texture height."""
        return self._attributes["MaxTexture2DLayeredHeight"]

    @property
    def max_texture2D_layered_layers(self):
        """Maximum layers in a 2D layered texture."""
        return self._attributes["MaxTexture2DLayeredLayers"]

    @property
    def max_texture2D_layered_width(self):
        """Maximum 2D layered texture width."""
        return self._attributes["MaxTexture2DLayeredWidth"]

    @property
    def max_texture2D_linear_height(self):
        """Maximum 2D linear texture height."""
        return self._attributes["MaxTexture2DLinearHeight"]

    @property
    def max_texture2D_linear_pitch(self):
        """Maximum 2D linear texture pitch in bytes."""
        return self._attributes["MaxTexture2DLinearPitch"]

    @property
    def max_texture2D_linear_width(self):
        """Maximum 2D linear texture width."""
        return self._attributes["MaxTexture2DLinearWidth"]

    @property
    def max_texture2D_mipmapped_height(self):
        """Maximum mipmapped 2D texture height."""
        return self._attributes["MaxTexture2DMipmappedHeight"]

    @property
    def max_texture2D_mipmapped_width(self):
        """Maximum mipmapped 2D texture width."""
        return self._attributes["MaxTexture2DMipmappedWidth"]

    @property
    def max_texture2D_width(self):
        """Maximum 2D texture width."""
        return self._attributes["MaxTexture2DWidth"]

    @property
    def max_texture3D_depth(self):
        """Maximum 3D texture depth."""
        return self._attributes["MaxTexture3DDepth"]

    @property
    def max_texture3D_depth_alt(self):
        """Alternate maximum 3D texture depth."""
        return self._attributes["MaxTexture3DDepthAlt"]

    @property
    def max_texture3D_height(self):
        """Maximum 3D texture height."""
        return self._attributes["MaxTexture3DHeight"]

    @property
    def max_texture3D_height_alt(self):
        """Alternate maximum 3D texture height."""
        return self._attributes["MaxTexture3DHeightAlt"]

    @property
    def max_texture3D_width(self):
        """Maximum 3D texture width."""
        return self._attributes["MaxTexture3DWidth"]

    @property
    def max_texture3D_width_alt(self):
        """Alternate maximum 3D texture width."""
        return self._attributes["MaxTexture3DWidthAlt"]

    @property
    def max_texture_cubemap_layered_layers(self):
        """Maximum layers in a cubemap layered texture."""
        return self._attributes["MaxTextureCubemapLayeredLayers"]

    @property
    def max_texture_cubemap_layered_width(self):
        """Maximum cubemap layered texture width/height."""
        return self._attributes["MaxTextureCubemapLayeredWidth"]

    @property
    def max_texture_cubemap_width(self):
        """Maximum cubemap texture width/height."""
        return self._attributes["MaxTextureCubemapWidth"]

    @property
    def max_threads_per_block(self):
        """
        The maximum number of threads per block, beyond which a launch
        of the function would fail. This number depends on both the function
        and the device on which the function is currently loaded.
        """
        return self._attributes["MaxThreadsPerBlock"]

    @property
    def max_threads_per_multi_processor(self):
        """Maximum resident threads per multiprocessor."""
        return self._attributes["MaxThreadsPerMultiProcessor"]

    @property
    def memory_clock_rate(self):
        """Typical clock frequency in kilohertz."""
        return self._attributes["MemoryClockRate"]

    @property
    def multi_gpu_board_group_id(self):
        """Unique id for a group of devices on the same multi-GPU board."""
        return self._attributes["MultiGpuBoardGroupID"]

    @property
    def multi_processor_count(self):
        """Number of multiprocessors on device."""
        return self._attributes["MultiProcessorCount"]

    @property
    def pageable_memory_access(self):
        """Device supports coherently accessing pageable memory without calling cudaHostRegister on it."""
        return bool(self._attributes["PageableMemoryAccess"])

    @property
    def pageable_memory_access_uses_host_page_tables(self):
        """Device accesses pageable memory via the host's page tables."""
        return bool(self._attributes["PageableMemoryAccessUsesHostPageTables"])

    @property
    def pci_bus_id(self):
        """PCI bus ID of the device."""
        return self._attributes["PciBusId"]

    @property
    def pci_device_id(self):
        """PCI device ID of the device."""
        return self._attributes["PciDeviceId"]

    @property
    def pci_domain_id(self):
        """PCI domain ID of the device."""
        return self._attributes["PciDomainId"]

    @property
    def can_use_stream_mem_ops(self):
        """cuStreamBatchMemOp and related APIs are supported."""
        return bool(self._attributes["Reserved92"])

    @property
    def can_use_64_bit_stream_mem_ops(self):
        """64-bit operations are supported in cuStreamBatchMemOp and related APIs."""
        return bool(self._attributes["Reserved93"])

    @property
    def can_use_stream_wait_value_nor(self):
        """CU_STREAM_WAIT_VALUE_NOR is supported."""
        return bool(self._attributes["Reserved94"])

    @property
    def single_to_double_precision_perf_ratio(self):
        """
        Ratio of single precision performance (in floating-point
        operations per second) to double precision performance.
        """
        return self._attributes["SingleToDoublePrecisionPerfRatio"]

    @property
    def stream_priorities_supported(self):
        """Device supports stream priorities."""
        return bool(self._attributes["StreamPrioritiesSupported"])

    @property
    def surface_alignment(self):
        """Alignment requirement for surfaces."""
        return self._attributes["SurfaceAlignment"]

    @property
    def tcc_driver(self):
        """Device is using TCC driver model."""
        return bool(self._attributes["TccDriver"])

    @property
    def texture_alignment(self):
        """Alignment requirement for textures."""
        return self._attributes["TextureAlignment"]

    @property
    def texture_pitch_alignment(self):
        """Pitch alignment requirement for textures."""
        return self._attributes["TexturePitchAlignment"]

    @property
    def total_constant_memory(self):
        """Memory available on device for __constant__ variables in a CUDA C kernel in bytes."""
        return self._attributes["TotalConstantMemory"]

    @property
    def unified_addressing(self):
        """Device shares a unified address space with the host."""
        return bool(self._attributes["UnifiedAddressing"])

    @property
    def warp_size(self):
        """Warp size in threads."""
        return self._attributes["WarpSize"]
