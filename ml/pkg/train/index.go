package train

import "sync"

type functionSet struct {
	values map[int]struct{}
	*sync.RWMutex
}

func newSet() functionSet {
	return functionSet{
		values: make(map[int]struct{}),
	}
}

func newSetFromRange(amount int) functionSet {
	s := functionSet{
		values: make(map[int]struct{}),
	}
	for i := 0; i < amount; i++ {
		s.values[i] = struct{}{}
	}
	return s
}

// Add adds an element to the set
func (set functionSet) Add(i int) {
	set.Lock()
	defer set.Unlock()
	set.values[i] = struct{}{}
}

// Remove safely removes an element
func (set functionSet) Remove(i int) {
	set.Lock()
	defer set.Unlock()
	delete(set.values, i)
}

// Contains checks if the element is in the set
func (set functionSet) Contains(i int) bool {
	set.RLock()
	defer set.RUnlock()
	_, ok := set.values[i]
	return ok
}

func (set functionSet) Items() []int {
	set.RLock()
	defer set.RUnlock()
	keys := make([]int, 0, len(set.values))
	for k := range set.values {
		keys = append(keys, k)
	}

	return keys
}

// functionIndex keeps track of the state of the functions
// deployed in an iteration, such as number of functions
// still running
type functionIndex struct {
	running  functionSet
	finished functionSet
}

func newIndex(amount int) functionIndex {
	idx := functionIndex{
		running:  newSet(),
		finished: newSetFromRange(amount),
	}

	return idx
}

func (idx functionIndex) Running() []int {
	return idx.running.Items()
}

func (idx functionIndex) Finished() []int {
	return idx.finished.Items()
}

func (idx functionIndex) Reset(amount int) {
	idx.finished = newSet()
	idx.running = newSetFromRange(amount)
}

func (idx functionIndex) Finish(funcId int) {
	idx.running.Remove(funcId)
	idx.finished.Add(funcId)
}
