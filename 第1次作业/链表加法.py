class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def list2linkedlist(nums):
    if not nums:
        return None
    head = ListNode(nums[0])
    ret = head
    for i in range(1, len(nums)):
        head.next = ListNode(nums[i])
        head = head.next
    return ret

def addLinkedList(head1, head2):
    ret = ListNode(0)
    head = ret
    carry = False # 当前位是否进位
    while head1 and head2:
        val = head1.val + head2.val + 1 if carry else head1.val + head2.val
        carry = True if val >= 10 else False
        head.next = ListNode(val%10)
        head = head.next
        head1 = head1.next
        head2 = head2.next
    if not head1 and not head2: # 两个数字位数相同
        if carry: # 要进位，即多加一位1
            head.next = ListNode(1)
    elif head1: # 两个数字位数不相同，第一个更长
        head.next = head1
        while carry and head1 and head1.val == 9: # 如果这一位是9且需要进位
            head1.val = 0
            if not head1.next:
                head1.next = ListNode(1)
                carry = False
                break
            else:
                head1 = head1.next
        head1.val += 1 if carry else 0 # 这一位不是9，可能需要进位
    elif head2: # 两个数字位数不相同，第二个更长
        head.next = head2
        while carry and head2 and head2.val == 9:
            head2.val = 0
            if not head2.next:
                head2.next = ListNode(1)
                carry = False
                break
            else:
                head2 = head2.next
        head2.val += 1 if carry else 0 # 这一位不是9，可能需要进位
        
    return ret.next

def printLinkedList(head):
    ret = ''
    while head:
        ret += str(head.val) + '->'
        head = head.next
    return ret[:-2]

num1 = list2linkedlist([2,4,3])
num2 = list2linkedlist([5,6,4])
print('Input:', printLinkedList(num1), ',', printLinkedList(num2))
print('Output:', printLinkedList(addLinkedList(num1, num2)))



































