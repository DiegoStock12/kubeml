package train

import "sync"

type functionSet struct {
	values map[int]struct{}
	*sync.RWMutex
}

// functionIndex keeps track of the state of the functions
// deployed in an iteration, such as number of functions
// still running
type functionIndex struct {
	running  int
	finished int
	*sync.Mutex
}

func newIndex(amount int) functionIndex {
	return functionIndex{
		running:  amount,
		finished: 0,
	}
}

func (idx functionIndex) reset(amount int) {
	idx.Lock()
	defer idx.Unlock()

	idx.finished = 0
	idx.running = amount
}

func (idx functionIndex) finish() {
	idx.Lock()
	defer idx.Unlock()

	idx.running--
	idx.finished++
}

// Add adds an element to the set
func (set functionSet) Add(i int) {
	set.Lock()
	defer set.Unlock()
	set.values[i] = struct{}{}
}

func (set functionSet) Remove(i int) {
	set.Lock()
	defer set.Unlock()
	delete(set.values, i)
}

func (set functionSet) Contains(i int) bool {
	set.RLock()
	defer set.RUnlock()
	_, ok := set.values[i]
	return ok
}
