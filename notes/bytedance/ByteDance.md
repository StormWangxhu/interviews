牛客网面经搜集的关于头条的算法题整理

* [排序](#排序)
    * [堆排序](#堆排序)
    
* [链表]()
    * [1.链表反转](#链表反转)
    * [2.判断一个链表是否是回文结构](#判断一个链表是否是回文结构)
    * [3.复制含有随机指针节点的链表](#复制含有随机指针节点的链表)
    * [4.单链表奇数位升许偶数位降序整体使有序(频率很高)](#单链表奇数位升许偶数位降序整体使有序)
    * [5.两个单链表相交的一系列问题]()
    * [6.删除单链表中重复的节点]()
    * [7.将搜索二叉树转换为双向链表]()
    * [8.合并两个有序的单链表](#合并两个有序的单链表)
* [二叉树](#二叉树)
    * [1.二叉树先中后序递归和非递归遍历](#二叉树先中后序遍历)
    * [2.树的路径和为n的路径(频率很高)](#树的路径和为n的路径)
    * [3.二叉树的直径](#二叉树的直径)
    * [4.二叉树的最长路径(频率很高)]()
    * [5.判断一个树是否是查找树](#判断一个树是否是查找树)
    * [6.二叉树的层次遍历(频率很高)](#二叉树的层次遍历)
    * [7.二叉查找树中查找与给定节点最近的节点]()
    * [8.二叉树转换成双向链表]()
* [字符串]()
    * [1.判断字符数组中是否所有的字符都只出现过一次]()
    * [2.找到字符串中的最长无重复字符子串]()
* [数组]()
    * [1.在行和列都排好序的矩阵中找数](#在行和列都排好序的矩阵中找数)
    * [2.奇数下标都是奇数偶数下标都是偶数](#奇数下标都是奇数偶数下标都是偶数)
    * [3.子数组的最大累加和问题](#子数组的最大累加和问题)
    * [4.边界都是1的最大正方形大小(频率很高leetcode85)]()
    * [5.有序数组被旋转过后,求最小点(leetcode153)](#有序数组被旋转过后,求最小点)
    * [6.找出一个有序数组数组的中位数(频率很高)<1>需要实现]()
    * [7.两个有序数组的中位数<2>需要实现](#寻找两个有序数组的中位数)
    * [8.两个无序数组的中位数<3>需要实现]()
* [大数据和空间限制]()
    * [1.认识布隆过滤器]()
    * [2.只用2G内存在20亿个整数中找到出现次数最多的数]()
    * [3.找到100亿个URL中重复的URL以及搜索词汇的topK问题]()
    * [4.一致性Hash算法的基本原理]() 
  
* [栈和队列](#栈和队列)
    * [1.用两个栈实现队列](#用两个栈实现队列)
    
* [其他]()
    * [1.N个数字,求出其中第K大的数]()
    * [2.滑动窗口问题(频率很高)]()
    * [3.设计RandomPool结构(频率很高)](#设计RandomPool结构)
    * [4.LRUCache缓存机制](#LRUCache缓存机制)
  
* [Linux](#Linux)
    * [top命令查看cpu/内存](#top命令查看cpu/内存)
    * [netstat命令查看网络/接口]()


# 数组


## 有序数组被旋转过后,求最小点

旋转概念: 把一个数组最开始的若干的元素搬到数组的末尾,我们称之为旋转.

[leetcode153.寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/description/)

### Solution

方法一:最日鬼的
```java
class Solution {
    
    public int findMin(int[] arr){
        Arrays.sort(arr);
        return arr[0];
    }
    
}
```

方法二:二分法

```java
class Solution {
    
    public int findMin(int[] arr) {
        
        if(arr==null||arr.length==0) 
            return 0;
        if(arr.length==1) 
            return arr[0];
        int L = 0,R = arr.length-1;
        while(L<R){
            int mid = L+(R-L)/2;
            if(arr[L]<=arr[mid]&&arr[L]>arr[R])
                L = mid+1;
            else
                R = mid;
        }
        return arr[L];
    }
}
```

## 子数组的最大累加和问题

[leetcode53.子数组的最大累加和问题](https://leetcode-cn.com/problems/maximum-subarray/description/)

### Solution

[github/StormWangxhu]()

方法一:时间复杂度:O(N),空间复杂度O(1)

```java
static class Solution {

        public int maxSubArray(int[] nums) {
            if (nums == null || nums.length == 0) {
                return 0;
            }

            int max = Integer.MIN_VALUE;
            int cur = 0;
            for (int i = 0; i != nums.length; i++) {
                cur += nums[i];
                max = Math.max(max, cur);
                cur = cur < 0 ? 0 : cur;
            }
            return max;
        }
    }
```

方法二:时间复杂度:O(N),空间复杂度:O(N)

```java
static class Solution1 {

        public int maxSubArray(int[] nums) {
            int temp = nums[0];
            int sum = nums[0];
            for (int i = 1; i < nums.length; i++) {
                if (temp > 0) {
                    temp += nums[i];
                } else {
                    temp = nums[i];
                }

                if (temp > sum)
                    sum = temp;
            }
            return sum;
        }
    }
```

## 奇数下标都是奇数偶数下标都是偶数

[leetcode922.按奇偶排序数组](https://leetcode-cn.com/problems/sort-array-by-parity-ii/description/)

### Solution

[github/StormWangxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/array/Question922.java)

```java
static class Solution{

        public int[] sortArrayByParityII(int[] A) {

            int[] sort = new int[A.length];
            int index0 = 0;
            int index1 = 1;
            for(int a: A){
                if(a % 2==0 ){//偶数
                    sort[index0] = a;
                    index0 += 2;
                }else { //奇数
                    sort[index1] = a;
                    index1 += 2;
                }
            }
            return sort;
        }
    }
```

## 在行和列都排好序的矩阵中找数

[leetcode74.搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/description/)

### Solution

[github/StormWangxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/array/Question74.java)

```java
static class Solution {

        public boolean searchMatrix(int[][] matrix, int target) {

            if (matrix.length == 0)
                return false;
            int row = 0, col = matrix[0].length - 1;

            while (row < matrix.length && col >= 0) {
                if (matrix[row][col] == target)
                    return true;
                if (matrix[row][col] < target)
                    row++;
                else
                    col--;
            }
            return false;
        }
    }
```

## 寻找两个有序数组的中位数

[leetcode04.两个有序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/description/)

### Solution

难度级别:难

[github/StormWangxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/array/Question04.java)

```java

    static class Solution {

        public double findMedianSortedArrays(int[] nums1, int[] nums2) {
            if (nums1.length > nums2.length)
                return findMedianSortedArrays(nums2, nums1); //ensure always nums1 is the shortest array
            int T = nums1.length + nums2.length, low = -1, high = -1;
            int median = (T - 1) / 2;
            boolean isOdd = false;
            if ((T % 2) != 0)
                isOdd = true;

            int s = 0, e = nums1.length - 1;
            while (s <= e) {
                int m = s + (e - s) / 2;
                if ((median - m - 1) < 0 || nums1[m] >= nums2[median - m - 1]) {
                    e = m - 1;
                    low = m;
                    high = median - m;
                } else s = m + 1;
            }

            if (low == -1) {
                if (isOdd) return nums2[median - nums1.length];
                else return (double) (nums2[median - nums1.length] + nums2[median - nums1.length + 1]) / 2.0D;
            } else {
                if (isOdd) return nums1[low] < nums2[high] ? nums1[low] : nums2[high];
                else {
                    //Always sorts maximum of 4 elements hence works in O(1)
                    List<Integer> list = new ArrayList<>();
                    list.add(nums1[low]);
                    if (low + 1 < nums1.length)
                        list.add(nums1[low + 1]);
                    list.add(nums2[high]);
                    if (high + 1 < nums2.length)
                        list.add(nums2[high + 1]);
                    Collections.sort(list);
                    return (double) (list.get(0) + list.get(1)) / 2.0;
                }
            }
        }
    }
```

# Linux 
 
 ## top命令查看cpu/内存
 
 查看cpu,内存

https://www.cnblogs.com/dragonsuc/p/5512797.html


 # 排序
 
 稳定性: 假设在数列中存在a[i]和a[j] ,若在排序前,a[i]在a[j]前面;并且在排序后,a[i]还是在a[j]前面,则这个算法就是稳定的!
 
 在八大排序中:
 
 插入,选择,冒泡:　　
 > 时间复杂度: O(N^2),空间复杂度:O(N),都具有稳定性
 
 归并,快排,堆排序:
 > 时间复杂度: O(NlogN)  
 > 空间复杂度: {堆:O(1),不稳定},{快排:O(logN),常规实现不稳定,但是可以做到具有稳定性,左神说比较难,不讲},{归并:O(N)具有稳定性}
 
 ##  堆排序
 
 堆排序讲解链接
 
 https://www.cnblogs.com/Java3y/p/8639937.html
 
 http://www.cnblogs.com/skywang12345/p/3602162.html#a43
 
 ### 堆简介
 堆排序（英语：Heapsort）是指利用堆这种数据结构所设计的一种排序算法。堆是一个近似完全二叉树的结构，并同时满足堆积的性质：即子结点的键值或索引总是小于（或者大于）它的父节点.
 ### 堆的操作
 在堆的数据结构中，堆中的最大值总是位于根节点（在优先队列中使用堆的话堆中的最小值位于根节点）。堆中定义以下几种操作：
 * 最大堆调整（Max Heapify）：将堆的末端子节点作调整，使得子节点永远小于父节点
 * 创建最大堆（Build Max Heap）：将堆中的所有数据重新排序
 * 堆排序（HeapSort）：移除位在第一个数据的根节点，并做最大堆调整的递归运算 
 
 ### 堆的性质
 
 当前节点索引 i,则:
 
 父节点索引:       (i-1)/2
 
 左孩子节点索引:   2*i+1
 
 右孩子节点索引：　 2*i+2
 
 大根堆：　父＞子
 
 小根堆：　父＜子
 
 ### 堆的Java实现

```java
public class HeapSort {

    public static void heapSort(int[] arr) {

        if (arr == null || arr.length < 2) {
            return;
        }
        
        //第一次建堆
        for (int i = 0; i < arr.length; i++) {
            heapInsert(arr, i);
        }
        int size = arr.length;//堆中元素的个数
        //第一次交换
        swap(arr, 0, --size);
        /**
         * 一直就是一个建堆和交换的过程
         * 建一次堆,交换一次(堆顶元素和堆最后一个元素交换,即数组第一位和最后一位交换),就完成一次排序
         */
        while (size > 0) {
            heapify(arr, 0, size);//建堆
            swap(arr, 0, --size);//交换
        }
    }

    public static void heapInsert(int[] arr, int index) {

        while (arr[index] > arr[(index - 1) / 2]) {
            swap(arr, index, (index - 1) / 2);
            index = (index - 1) / 2;
        }
    }

    /**
     * 在构建最大堆
     *
     * @param arr
     * @param index
     * @param size
     */
    public static void heapify(int[] arr, int index, int size) {

        int left = index * 2 + 1;
        // int right = index *2 +2 ;
        while (left < size) {
            /**
             * left+1表示右节点
             * 左节点和右节点比较,得出左右节点当中的最大的节点
             * 然后再和左右节点的根节点比较得出三个中的最大值
             *
             */
            int largest = left + 1 < size && arr[left + 1] > arr[left] ? left + 1 : left;//得出左右节点最大的
            largest = arr[largest] > arr[index] ? largest : index; //和根节点比较得出最大的
            if (largest == index) {
                break;
            }
            swap(arr, largest, index); //根节点进行交换
            index = largest;
            left = index * 2 + 1;
        }
    }

    public static void swap(int[] arr, int i, int j) {

        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    
}
```


# 链表

## 复制含有随机指针的链表

[leetcode138.复制含有随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/description/)


### Solution

```java
  static class Solution{

        public RandomListNode copyRandomList(RandomListNode head) {

            if (head == null) return null;
            RandomListNode ite = head, next;
            while (ite != null) {
                next = ite.next;
                RandomListNode node = new RandomListNode(ite.label);
                ite.next = node;
                node.next = next;
                ite = next;
            }

            ite = head;
            while (ite != null) {
                if (ite.random != null)
                    ite.next.random = ite.random.next;
                ite = ite.next.next;
            }

            ite = head;
            RandomListNode copyIte = ite.next, copyHead = ite.next;
            while (copyIte.next != null) {
                ite.next = copyIte.next;
                copyIte.next = ite.next.next;
                copyIte = copyIte.next;
                ite = ite.next;
            }
            ite.next = null;

            return copyHead;

        }
    }
```
## 翻转单向链表双向链表

[leetcood206.反转单向链表](https://leetcode-cn.com/problems/reverse-linked-list/description/)

### Solution

单向链表
```java
public class Question206 {

    static class Solution {

        public ListNode reverseList(ListNode head) {

            if (head == null) {
                return head;
            }

            ListNode pre = null;
            ListNode next = null;

            while (head != null) {
                next = head.next;//将第二节点进行保存
                head.next = pre; //头结点指针指向前一个节点
                pre = head; //pre指针向后移
                head = next; //头结点从新定义
            }
            return pre;
        }
    }
}

```
双向链表 和反转单向链表一样,改变指针方向
```java
static class Solution1 {

        public DoubleNode reverseList(DoubleNode head) {

            if (head == null) {
                return head;
            }

            DoubleNode pre = null;
            DoubleNode next = null;
            while (head != null) {
                next = head.next;
                head.next = pre;
                head.last = next;
                pre = head;
                head = next;
            }
            return pre;
        }
    }
```

## 反转部分单向链表

[leetcode92.反转部分链表](https://leetcode-cn.com/problems/reverse-linked-list-ii/description/)

### Solution

```java
static class Solution {

        public ListNode reverseBetween(ListNode head, int m, int n) {

            int len = 0;
            ListNode node1 = head;
            ListNode fPre = null;
            ListNode tPos = null;
            while (node1 != null) {
                len++;
                fPre = len == m-1 ? node1 : fPre;
                tPos = len == n+1 ? node1 : tPos;
                node1 = node1.next;
            }

            if (m > n || m < 0 || n > len) {
                return head;
            }

            /**
             * 此处部分相当于反转单向链表那中逻辑
             */
            //node1现在是我要反转部分的头节点
            node1 = fPre == null ? head : fPre.next;
            ListNode node2 = node1.next;
            node1.next = tPos;
            ListNode next = null;
            while (node2 != tPos) {
                next = node2.next;
                node2.next = node1;
                node1 = node2;
                node2 = next;
            }

            if (fPre != null) {
                fPre.next = node1;
                return head;
            }
            return node1;
        }
    }
```


# 判断一个链表是否是回文结构

[leetcode234.回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/description/)

## Solution

方法一思路: 利用栈结构,从左到右遍历链表,遍历的过程中把每个节点依次压入栈中.因为栈的先进后出特点,所以在遍历完成后,从栈顶到栈底的节点值出现顺序会与原链表从左到右出现顺序反过来.那么,如果一个链表是回文结构,逆序后,值出现的次序还是一样的,若不是回文结构,顺序肯定对不上.
```java
class Solution {
    
        public boolean isPalindrome(ListNode head) {
            
           Stack<ListNode> stack = new Stack<>();
            ListNode cur = head;

            while (cur != null) {
                stack.push(cur);
                cur = cur.next;
            }

            while (head != null) {
                if (head.val != stack.pop().val) {
                    return false;
                }
                head = head.next;
            }
            return true;
        }
}
```
还有方法二:也是利用栈结构,但其实并不需要将所有节点都压入栈中,只用压入一半的节点即可.首先假设链表的长度的为N,如果N为偶数,前N/2的节点为左半区,后N/2的节点为右半区.如果N是奇数,忽略位于中间的节点,还是前N/2为左半区,右N/2为右半区.


方法三:

```java
class Solution {
    
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) return true;
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        if (fast != null) slow = slow.next;  // 偶数节点，让 slow 指向下一个节点
        cut(head, slow);                     // 切成两个链表
        return isEqual(head, reverse(slow));
    }

    private void cut(ListNode head, ListNode cutNode) {
        while (head.next != cutNode) {
            head = head.next;
        }
        head.next = null;
    }

    private ListNode reverse(ListNode head) {
        ListNode newHead = null;
        while (head != null) {
            ListNode nextNode = head.next;
            head.next = newHead;
            newHead = head;
            head = nextNode;
        }
        return newHead;
    }

    private boolean isEqual(ListNode l1, ListNode l2) {
        while (l1 != null && l2 != null) {
            if (l1.val != l2.val) return false;
            l1 = l1.next;
            l2 = l2.next;
        }
        return true;
    }
}
```


# 合并两个有序的单链表

[leetcode21.合并两个有序的单链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/description/)

## Solution

```java
static class Solution{

        public ListNode mergeTwoLists(ListNode l1, ListNode l2) {

            if (l1==null){
                return l2;
            }

            if (l2==null){
                return l1;
            }

            if (l1.val<l2.val){
                l1.next = mergeTwoLists(l1.next,l2);
                return l1;
            }else {
                l2.next = mergeTwoLists(l1,l2.next);
                return l2;
            }
        }
}
```

# 单链表奇数位升许偶数位降序整体使有序

https://www.cnblogs.com/DarrenChan/p/8764608.html

https://blog.csdn.net/zxm1306192988/article/details/82837470


 
# 删除有序单链表中重复节点

https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/linkedlist/Question83.java

# 其他

## 设计RandomPool结构

[leetcode381.O(1) 时间插入、删除和获取随机元素 - 允许重复](https://leetcode-cn.com/problems/insert-delete-getrandom-o1-duplicates-allowed/description/)

```java
static class RandomizedCollection {

        private Map<Integer, Set<Integer>> map;
        private List<Integer> list;

        /**
         * Initialize your data structure here.
         */
        public RandomizedCollection() {
            map = new HashMap<>();
            list = new ArrayList<>();
        }

        /**
         * Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
         */
        public boolean insert(int val) {
            boolean status = map.containsKey(val);
            Set<Integer> set = map.get(val);
            if (set == null) {
                set = new HashSet<>();
                map.put(val, set);
            }
            list.add(val);
            set.add(list.size() - 1);
            return !status;
        }

        /**
         * Removes a value from the collection. Returns true if the collection contained the specified element.
         */
        public boolean remove(int val) {
            if (map.containsKey(val)) {
                Set<Integer> set = map.get(val);
                int valIndex = set.iterator().next();
                set.remove(valIndex);
                if (set.isEmpty()) {
                    map.remove(val);
                }
                if (valIndex == list.size() - 1) { //if this is the last index then simply remove it
                    list.remove(list.size() - 1);
                } else {
                    int lastEle = list.get(list.size() - 1);
                    map.get(lastEle).remove(list.size() - 1);
                    map.get(lastEle).add(valIndex);
                    list.set(valIndex, lastEle);
                    list.remove(list.size() - 1);
                }
                return true;
            } else return false;
        }

        /**
         * Get a random element from the collection.
         */
        public int getRandom() {
            Random random = new Random();
            return list.get(random.nextInt(list.size()));
        }

    }
```


## LRUCache缓存机制

[leetcode146.LRUCache缓存机制](https://leetcode-cn.com/problems/lru-cache/description/)


```java
static class LRUCache{

        public static class DLinkList {
            int key, value;
            DLinkList left;
            DLinkList right;

            DLinkList(int key, int value) {
                this.key = key;
                this.value = value;
                left = null;
                right = null;
            }
        }

        private Map<Integer, DLinkList> cache;
        private DLinkList head, tail;
        private int capacity, currentSize;

        /**
         * Pop head node
         *
         * @return
         */
        private DLinkList popHead() {
            if (!head.right.equals(tail)) {
                DLinkList node = head.right;
                head.right = node.right;
                node.right.left = head;
                node.right = null;
                node.left = null;
                return node;
            }
            return null;
        }

        /**
         * Push to tail
         *
         * @param node
         */
        private void offer(DLinkList node) {
            tail.left.right = node;
            node.left = tail.left;
            node.right = tail;
            tail.left = node;
        }

        /**
         * Move node to tail
         *
         * @param node
         */
        private void moveToTail(DLinkList node) {
            node.left.right = node.right;
            node.right.left = node.left;
            offer(node);
        }

        /**
         * Main method
         *
         * @param args
         * @throws Exception
         */
        public static void main(String[] args) throws Exception {
            LRUCache cache = new LRUCache(2);
            cache.put(1, 1);
            cache.put(2, 2);
            System.out.println(cache.get(1));
            cache.put(3, 3);
            System.out.println(cache.get(2));
            cache.put(4, 4);
            System.out.println(cache.get(1));
            System.out.println(cache.get(3));
            System.out.println(cache.get(4));
        }

        public LRUCache(int capacity) {
            this.capacity = capacity;
            this.currentSize = 0;
            cache = new HashMap<>();
            head = new DLinkList(-1, -1);
            tail = new DLinkList(-1, -1);
            head.right = tail;
            tail.left = head;
        }

        public int get(int key) {
            if (cache.get(key) == null) return -1;
            DLinkList node = cache.get(key);
            moveToTail(node);
            return node.value;
        }

        public void put(int key, int value) {
            if (cache.containsKey(key)) {
                DLinkList node = cache.get(key);
                node.value = value;
                moveToTail(node);
            } else {
                if (capacity == currentSize) {
                    DLinkList head = popHead();
                    if (head != null) {
                        cache.remove(head.key);
                        DLinkList node = new DLinkList(key, value);
                        offer(node);
                        cache.put(key, node);
                    }
                } else {
                    DLinkList node = new DLinkList(key, value);
                    offer(node);
                    cache.put(key, node);
                    ++currentSize;
                }
            }
        }
    }
```



# 二叉树

## 二叉树先中后序遍历

[Leetcode144.二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/description/)

[leetcode94.二叉树中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/description/)

[leetcode145.二叉树后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/description/)

### Solution

1.二叉树先序遍历 递归版

[github/StormWangxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/tree/recursive/Question144.java)

```java
   /**
     * 递归实现
     */
    static class Solution {

        static List<Integer> res;

        public List<Integer> preorderTraversal(TreeNode root) {
            res = new ArrayList<>();
            preOrderTraversal(root);
            return res;
        }

        private void preOrderTraversal(TreeNode root) {
            if (root == null) {
                return;
            }

            res.add(root.val);
            preOrderTraversal(root.left);
            preOrderTraversal(root.right);
        }
    }
```
1.二叉树先序遍历 迭代实现

```java
    /**
     * 非递归实现二叉树的前序遍历
     * 数据结构:栈
     * 写法二
     */
    static class Solution3 {

        public List<Integer> preorderTraversal(TreeNode root) {

            List<Integer> list = new ArrayList<>();
            Stack<TreeNode> stack = new Stack<>();
            stack.add(root);
            while (!stack.isEmpty()) {
                TreeNode node = stack.pop();
                if (node == null) {
                    continue;
                }
                list.add(node.val);
                stack.push(node.right);//先右后左,出栈的时候就会是先左后右
                stack.push(node.left);
            }
            return list;
        }
    }
```

2.二叉树中序遍历 递归版

[github/StormWangxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/tree/recursive/Question94.java)

```java
static class Solution {

        public static List<Integer> res;

        public List<Integer> inorderTraversal(TreeNode root) {
            res = new ArrayList<>();
            inOrderTraversal(root);
            return res;
        }

        private void inOrderTraversal(TreeNode root) {
            if (root == null) {
                return;
            }
            inOrderTraversal(root.left);
            res.add(root.val);
            inOrderTraversal(root.right);
        }
    }
```
2.二叉树中序遍历 迭代实现
```java
   static class Solution1 {

        public static List<Integer> res;
        public List<Integer> inorderTraversal(TreeNode root) {
            res = new ArrayList<>();
            inOrderTraversal(root);
            return res;
        }

        private void inOrderTraversal(TreeNode root) {
            if (root == null) {
                return;
            }
            Stack<TreeNode> stack = new Stack<>();
            while (!stack.isEmpty() || root != null) {
                if (root != null) {
                    stack.push(root);
                    root = root.left;
                } else {
                    root = stack.pop();
                    res.add(root.val);
                    root = root.right;
                }
            }
            System.out.println();
        }
    }
```
3.二叉树后序遍历 递归实现

[github/StormWangxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/tree/recursive/Question145.java)

```java
static class Solution1 {

        private List<Integer> res;

        public List<Integer> postorderTraversal(TreeNode root) {
            res = new ArrayList<>();
            if (root == null) {
                return res;
            }
            postOrderTraversal(root);
            return res;
        }

        private void postOrderTraversal(TreeNode root) {
            if (root == null) {
                return;
            }
            postOrderTraversal(root.left);
            postOrderTraversal(root.right);
            res.add(root.val);
        }
    }
```

二叉树后续遍历 迭代实现
```java
static class Solution {

        public List<Integer> postorderTraversal(TreeNode root) {

            List<Integer> list = new ArrayList<>();
            Stack<TreeNode> stack = new Stack<>();
            if (root == null) {
                return list;
            }
            stack.push(root);
            while (!stack.isEmpty()) {
                root = stack.pop();
                if (root == null) {
                    continue;
                }
                list.add(root.val);
                stack.push(root.left);
                stack.push(root.right);
            }
            Collections.reverse(list);
            return list;
        }
    }
```



## 判断一个树是否是查找树

[leetcode98.验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/description/)

[github/StormWangxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/tree/BST/Question98.java)

### Solution

```java
static class Range{
        long low, high;
    }

    static class Solution {

        public boolean isValidBST(TreeNode root) {

            if (root == null ||
                    (root.right == null && root.left == null)) return true;
            Range range = new Range();
            range.high = Long.MAX_VALUE;
            range.low = Long.MIN_VALUE;
            return validate(root, range);

        }

        private boolean validate(TreeNode root, Range range) {

            if ((root.val > range.low) && (root.val < range.high)) {
                long temp = range.high;
                if (root.left != null) {
                    range.high = root.val;
                    if (!validate(root.left, range)) return false;
                }
                if (root.right != null) {
                    range.high = temp;
                    range.low = root.val;
                    if (!validate(root.right, range)) return false;
                }
                return true;
            } else
                return false;
        }
    }

```


## 树的路径和为n的路径

[leetcode437.路径总和III](https://leetcode-cn.com/problems/path-sum-iii/description/)

[github/StormWangxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/tree/recursive/Question437.java)


### Solution

```java
class Solution {
    
        public int pathSum(TreeNode root, int sum) {

            if (root == null) {
                return 0;
            }
            int res = pathSumWithRoot(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
            return res;
        }

        private int pathSumWithRoot(TreeNode root, int sum) {
             if (root == null) {
                return 0;
            }
            int res = 0;
            if (root.val == sum) {
                res++;
            }
            res += pathSumWithRoot(root.left, sum - root.val) + pathSumWithRoot(root.right, sum - root.val);
            return res;
        }
}
```


## 二叉树的直径

[leetcode543.二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/description/)

[github/StormWngxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/tree/recursive/Question543.java)

### Solution

```java
static class Solution {

        private int max = 0;
        public int diameterOfBinaryTree(TreeNode root) {
            maxDepth(root);
            return max;
        }

        private int maxDepth(TreeNode root) {
            if (root == null) {
                return 0;
            }
            int leftDepth = maxDepth(root.left);
            int rightDepth = maxDepth(root.right);
            max = Math.max(max, (leftDepth + rightDepth));
            return Math.max(leftDepth, rightDepth) + 1;
        }
    }
```

## 二叉树的层次遍历

[leetcode102.二叉树的层次遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/description/)

[github/StormWangxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/tree/bfs/Question102.java)

### Solution

```java
    static class Solution {

        public List<List<Integer>> levelOrder(TreeNode root) {

            List<List<Integer>> lists = new ArrayList<>();
            Queue<TreeNode> queue = new LinkedList<>();

            if (root == null) {
                return lists;
            }
            queue.add(root);
            while (!queue.isEmpty()) {
                int size = queue.size();
                List<Integer> subList = new ArrayList<>();
                while (size-- > 0) {
                    root = queue.poll();
                    subList.add(root.val);
                    if (root.left != null) {
                        queue.add(root.left);
                    }
                    if (root.right != null) {
                        queue.add(root.right);
                    }
                }
                lists.add(subList);
            }
            return lists;
        }
    }
```

# 栈和队列

## 用两个栈实现队列

[leetcode232.用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/description/)

[github/StormWangxhu](https://github.com/StormWangxhu/algorithm/blob/master/src/me/wangxhu/leedcode/stack/Question232.java)

### Solution

用两个栈实现队列,一个做数据压入栈,一个做数据弹出栈,注意两点:数据压入栈中的数据要一次性全部弹出;数据弹出栈不为空的时候我们不能压入数据.

```java
static class MyQueue {

        private Stack<Integer> stackPush;
        private Stack<Integer> stackPop;

        public MyQueue() {
            stackPush = new Stack<>();
            stackPop = new Stack<>();
        }


        public void push(int x) {
            stackPush.push(x);
        }

        public int pop() {
            if (stackPush.isEmpty() && stackPop.isEmpty()) {
                throw new RuntimeException("stack is empty");
            } else if (stackPop.isEmpty()) {
                while (!stackPush.isEmpty()) {
                    stackPop.push(stackPush.pop());
                }
            }
            return stackPop.pop();
        }


        public int peek() {
            if (stackPush.isEmpty() && stackPop.isEmpty()) {
                throw new RuntimeException("stack is empty");
            } else if (stackPop.isEmpty()) {
                while (!stackPush.isEmpty()) {
                    stackPop.push(stackPush.pop());
                }
            }
            return stackPop.peek();
        }

        public boolean empty() {
            if (stackPop.isEmpty() && stackPush.isEmpty()) {
                return true;
            } else {
                return false;
            }
        }
        
    }
```