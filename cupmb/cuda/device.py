#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package pmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import cupy as cp

from pmb.utils.units import bytes2str
from pmb.cuda.device_attributes import CudaDeviceAttributes


class CudaDevice:
    """A CudaDevice wraps matching cupy device in a single object."""

    def __init__(self, device_id):
        """Initialize a cupy device from an integer device index."""
        self._cupy_device: cp.cuda.Device = cp.cuda.Device(device_id)
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        attrs = self._cupy_device.attributes
        self._device_attrs = CudaDeviceAttributes(attrs, props)

    @classmethod
    def from_pci_bus_identifier(cls, pci_bus_identifier: str):
        """Create a device directly from a full pci bus identifier."""
        device_id = cp.cuda.Device.from_pci_bus_id(pci_bus_identifier).id
        return cls(device_id)

    @property
    def cupy_handle(self):
        """Returns corresponding cupy device handle."""
        return self._cupy_device

    @property
    def attrs(self):
        """Returns the device attributes associated to this device."""
        return self._device_attrs

    @property
    def id(self):
        """Returns the device id."""
        return int(self._cupy_device.id)

    @property
    def compute_capability(self):
        """Returns the compute capability of this device as a tuple of (major, minor) integers."""
        return (self._device_attrs.compute_major, self._device_attrs.compute_minor)

    @property
    def architecture(self):
        """Returns the compute capability of this device as a string."""
        return self._cupy_device.compute_capability

    @property
    def name(self):
        """Returns device name."""
        return self._device_attrs.name

    @property
    def mem_info(self):
        """Returns the amount of total and free memory of the device, in bytes."""
        free, total = self._cupy_device.mem_info
        return (int(free), int(total))

    @property
    def pci_bus_identifier(self):
        """
        Returns PCI bus identifier string for the device in the following format
        [domain]:[bus]:[device].[function] where domain, bus, device, and function are
        all hexadecimal values. Domain, bus and device ids can be accessed with
        pci_domain_id, pci_bus_id and pci_device_id respectively.
        """
        return str(self._cupy_device.pci_bus_id)

    @staticmethod
    def cuda_runtime_version():
        """Get Cuda runtime version as an integer."""
        return int(cp.cuda.runtime.runtimeGetVersion())

    @staticmethod
    def cuda_driver_version():
        """Get Cuda driver version as an integer."""
        return int(cp.cuda.runtime.driverGetVersion())

    def use(self):
        """
        Makes this device current.
        If you want to switch a device temporarily, use the with statement.
        """
        self._cupy_device.use()

    def __enter__(self):
        """Set this device as the current device for cupy."""
        self._cupy_device.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous active device for cupy."""
        self._cupy_device.__exit__()

    def __str__(self):
        """Returns a short string identifying this device."""
        return f"({self.name})[id={self.id}, pci_bus_id={self.pci_bus_identifier}]"

    def long_description(self):
        """Returns a long string identifying this device."""
        m = self.mem_info
        return f"""CudaDevice[id={self.id}, pci_bus_id={self.pci_bus_identifier}, uuid={self._device_attrs.uuid.hex()}]
  Compute capability {'.'.join(map(str,self.device_capability))}, sm_{self.architecture} architecture
  Memory info: {bytes2str(m[0])} (free), {bytes2str(m[1])} (total)"""


class CudaDeviceGroup:
    """A dict-like object containing CudaDevice indexed by their device identifier."""

    def __init__(self):
        """Initialize an empty CudaDeviceGroup."""
        self._devices = {}

    @property
    def devices(self):
        """Returns devices contained in this device group as an OrderedDict."""
        return self._devices

    def insert_device(self, key, device):
        """Insert a device to device group."""
        if key in self._devices:
            msg = f"Key '{key}' already corresponds to a device."
            raise KeyError(msg)
        self._devices[key] = device
        return self

    def remove_device(self, key):
        """Remove a device from device group."""
        if key not in self._devices:
            msg = f"Key '{key}' do not corresponds to any device."
            raise KeyError(msg)
        del self._devices[key]
        return self

    def __getitem__(self, key):
        """Get the device that has been regisetered with given key."""
        if key in self._devices:
            return self._devices[key]
        else:
            msg = f"Unknown device key '{key}'."
            raise KeyError(msg)

    def __len__(self):
        """Return the number of devices contained in this group."""
        return len(self._devices)

    def __iter__(self):
        """Iterate over contained devices."""
        return iter(self._devices.values())

    def __str__(self):
        """Short string describing this object."""
        return f"CudaDeviceGroup containing {len(self)} devices"
