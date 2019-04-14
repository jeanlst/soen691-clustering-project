import numpy as np
import math

class BinHeap:
    """
    Binary Heap implementation
    Heap assumes elements are lists,
    where the first element is checked.
    Assumed min heap.
    """

    def __init__(self):
        #the heap proper, represented as a list
        self.__heap = []
        self.__size = 0

    def __get_children(self, i):
        """
        given a heap index, return it's "children"
        """

        child1 = 2 * i + 1
        child2 = 2 * i + 2

        return child1, child2

    def __min_child(self, i):
        """
        return the min child of a given index,
        or itself, if there are none
        """

        #if first child index less than size
        if (i * 2) + 1 < self.__size:
            
            min_c = (i * 2) + 1

            if (i * 2) + 2 < self.__size:

                c_2 = (i * 2) + 1

                #get values for both minimums
                v1 = self.__heap[min_c][0]
                v2 = self.__heap[c_2][0]

                #take the minimum
                if v2 < v1:
                    min_c = c_2
        else:

            #just return i
            min_c = i

        return min_c

    def __get_parent(self, i):
        """
        given a heap index, return it's parent
        """

        parent = math.floor((i - 1) / 2.0)

        return parent

    def __swapUp(self, ind):
        """
        reorder min heap after insertion
        """

        #while index in question is greater than zero
        while ind > 0:
            #get it's parent
            par = self.__get_parent(ind)
            #evaluate which is smaller
            i_val = self.__heap[ind][0]
            p_val = self.__heap[par][0]
            #print("index ", ind, " value ", i_val, " vs. parent", par, " value ", p_val)
            #if parent is larger, swap and set parent as index
            if p_val > i_val:
                #swap contents
                temp_v = self.__heap[par]
                self.__heap[par] = self.__heap[ind]
                self.__heap[ind] = temp_v

                #update ind
                ind = par
            else:
                #order set, just break
                break

    def __swapDown(self, ind):
        """
        after removal from top of list, swap down.
        check down from ind
        """

        #now go downwards
        stop_loop = False

        while stop_loop == False:

            #get minimum child for current
            mch = self.__min_child(ind)

            #if no children, stop
            if mch == ind:
                break
            else:
                #get value of minimum child and parent
                v_ind = self.__heap[ind][0]
                v_mch = self.__heap[mch][0]

                if v_mch < v_ind:
                    #swap values
                    temp_v = self.__heap[mch]
                    self.__heap[mch] = self.__heap[ind]
                    self.__heap[ind] = temp_v

                    #update index
                    ind = mch
                else:
                    #we're good, stop
                    break

    def insert(self, element):
        """
        Insert element into the heap
        """
        #add the element proper
        self.__size += 1
        self.__heap.append(element)

        #swap up
        self.__swapUp(self.__size - 1)

    def remove(self):
        """
        Remove top element from the heap
        """

        #remove top element, replace with last in list
        top_ele = self.__heap[0]
        last_ele = self.__heap.pop()
        self.__size -= 1
        if self.__size == 0:
            self.__heap.append(last_ele)
        else:
            self.__heap[0] = last_ele

        #swap down
        self.__swapDown(0)

        return top_ele

    def top(self):
        """
        Return top element of Heap
        """

        return self.__heap[0]

    def heapSize(self):
        """
        return number of elements in heap
        """

        return self.__size

    def print_heap(self):
        """
        Print heap contents
        """

        print(self.__heap)



if __name__ == "__main__":

    print("Test")

    #create heap
    heap = BinHeap()

    heap.print_heap()

    heap.insert([3,4])

    heap.print_heap()

    heap.insert([1,5])

    heap.print_heap()

    heap.insert([4,2])

    heap.print_heap()

    heap.insert([2,1])

    heap.print_heap()

    heap.remove()

    heap.print_heap()

    heap.remove()

    heap.print_heap()
