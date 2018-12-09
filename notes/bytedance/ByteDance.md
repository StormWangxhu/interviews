牛客网面经搜集的关于头条的算法题整理

* [排序](#排序)
    * [堆排序](#堆排序)
    
* [链表]()
    * [1.链表反转](#链表反转)
    * [2.判断一个链表是否是回文结构](#判断一个链表是否是回文结构)
    * [3.复制含有随机指针节点的链表](#)
    * [4.单链表奇数位升许偶数位降序整体使有序(频率很高)](#)
    * [5.两个单链表相交的一系列问题]()
    * [6.删除无序单链表中重复的节点]()
    * [7.将搜索二叉树转换为双向链表]()
    * [8.合并两个有序的单链表](#合并两个有序的单链表)
* [二叉树]()
    * [1.二叉树先中后序递归和非递归遍历]()
    * [2.树的路径和为n的路径(频率很高)]()
    * [3.二叉树的直径]()
    * [4.二叉树的最长路径(频率很高)]()
    * [5.判断一个树是否是查找树]()
    * [6.二叉树的层次遍历(频率很高)]()
    * [7.二叉查找树中查找与给定节点最近的节点]()
    * [8.二叉树转换成双向链表]()
* [字符串]()
    * [1.判断字符数组中是否所有的字符都只出现过一次]()
    * [2.找到字符串中的最长无重复字符子串]()
* [数组]()
    * [1.在行和列都排好序的矩阵中找数]()
    * [2.奇数下标都是奇数偶数下标都是偶数]()
    * [3.子数组的最大累加和问题]()
    * [4.边界都是1的最大正方形大小(频率很高)]()
    * [5.有序数组被旋转过后,求最小点]()
    * [6.找出一个有序数组数组的中位数(频率很高)<1>需要实现]()
    * [7.两个有序数组的中位数<2>需要实现]()
    * [8.两个无序数组的中位数<3>需要实现]()
* [大数据和空间限制]()
    * [1.认识布隆过滤器]()
    * [2.只用2G内存在20亿个整数中找到出现次数最多的数]()
    * [3.找到100亿个URL中重复的URL以及搜索词汇的topK问题]()
    * [4.一致性Hash算法的基本原理]() 
  
* [栈和队列]()
    * [1.用两个栈实现队列]()
    
* [其他]()
    * [1.N个数字,求出其中第K大的数]()
    * [2.滑动窗口问题(频率很高)]()
    * [3.设计RandomPool结构(频率很高)]()
  
 
* [Linux top命令](#Linux top命令)

 # top命令
 
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


# 链表反转

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


# 判断回文链表

[leetcode234.回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/description/)

## Solution

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
 