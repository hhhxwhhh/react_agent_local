"""二分查找。"""


def lower_bound(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if not nums or nums[lo] < target:
        return -1
    return lo
