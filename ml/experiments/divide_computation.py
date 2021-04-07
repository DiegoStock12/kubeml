import torch


def calculate_device(id, num_devices) -> int:
    return id % num_devices


if __name__ == '__main__':
    # get the number of devices
    device_num = torch.cuda.device_count()
    print(device_num)

    funcs = range(0, 8)
    devices = 2

    for f in funcs:
        print('function', f, 'goes to device', calculate_device(f, devices))
