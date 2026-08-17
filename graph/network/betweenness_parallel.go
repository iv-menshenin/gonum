package network

import (
	"context"
	"runtime"
	"sync"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/internal/linear"
)

// BetweennessParallel returns the non-zero betweenness centrality for nodes in the unweighted graph g.
//
//	C_B(v) = \sum_{s ≠ v ≠ t ∈ V} (\sigma_{st}(v) / \sigma_{st})
//
// where \sigma_{st} and \sigma_{st}(v) are the number of shortest paths from s to t,
// and the subset of those paths containing v respectively.
func BetweennessParallel(ctx context.Context, g graph.Graph) map[int64]float64 {
	// cb хранит итоговые значения центральности по посредничеству для каждого узла.
	// Ключ - ID узла, значение - его центральность.
	cb := make(map[int64]float64)
	var mu sync.Mutex // синхронизация доступа к cb

	// Функция accumulate вызывается для каждой исходной вершины s.
	// Она вычисляет вклады в cb, основанные на путях, исходящих из s.
	brandes_parallel(ctx, g, func(s graph.Node, stack linear.NodeStack, p map[int64][]graph.Node, delta, sigma map[int64]float64) {
		sID := s.ID()
		// Чтобы не блочить на каждое сложение, храним вклады от s тут
		sNodeContributions := make(map[int64]float64)

		// от дальних к ближним
		for stack.Len() != 0 {
			w := stack.Pop()
			wID := w.ID()
			for _, vPred := range p[wID] { // vPred - это предшественник вершины w на кратчайшем пути от s.
				vPredID := vPred.ID()
				if sigma[wID] != 0 { // если до w нет путей от s
					// delta_s(vPred) += (sigma_s(vPred) / sigma_s(w)) * (1 + delta_s(w))
					delta[vPredID] += (sigma[vPredID] / sigma[wID]) * (1 + delta[wID])
				}
			}
			if wID != sID {
				// аккумулируем для w: cb(w) += delta_s(w)
				if dVal := delta[wID]; dVal != 0 {
					sNodeContributions[wID] += dVal // сначала во временную, чтобы не блочить на каждое сложение
				}
			}
		}

		if len(sNodeContributions) > 0 {
			mu.Lock()
			for nodeID, val := range sNodeContributions {
				cb[nodeID] += val
			}
			mu.Unlock()
		}
	})
	return cb
}

// EdgeBetweennessParallel returns the non-zero betweenness centrality for edges in the
// unweighted graph g. For an edge e the centrality C_B is computed as
//
//	C_B(e) = \sum_{s ≠ t ∈ V} (\sigma_{st}(e) / \sigma_{st}),
//
// where \sigma_{st} and \sigma_{st}(e) are the number of shortest paths from s
// to t, and the subset of those paths containing e, respectively.
//
// If g is undirected, edges are retained such that u.ID < v.ID where u and v are
// the nodes of e.
func EdgeBetweennessParallel(ctx context.Context, g graph.Graph) map[[2]int64]float64 {
	_, isUndirected := g.(graph.Undirected)
	cb := make(map[[2]int64]float64)
	var mu sync.Mutex // синхронизация доступа к cb

	brandes_parallel(ctx, g, func(s graph.Node, stack linear.NodeStack, p map[int64][]graph.Node, delta, sigma map[int64]float64) {
		// Чтобы не блочить на каждое сложение, храним вклады от s тут
		sEdgeContributions := make(map[[2]int64]float64)

		// от дальних к ближним
		for stack.Len() != 0 {
			w := stack.Pop()
			wID := w.ID()
			for _, vPred := range p[wID] { // vPred - это предшественник вершины w на кратчайшем пути от s.
				vPredID := vPred.ID()
				var c float64 // вклад ребра (vPred, w) в его центральность для путей, исходящих от s
				if sigma[wID] != 0 {
					// c_s(vPred, w) = (sigma_s(vPred) / sigma_s(w)) * (1 + delta_s(w))
					c = (sigma[vPredID] / sigma[wID]) * (1 + delta[wID])
				}

				uid, vid := vPredID, wID
				if isUndirected && vid < uid {
					uid, vid = vid, uid
				}
				edgeKey := [2]int64{uid, vid}

				if c != 0 {
					sEdgeContributions[edgeKey] += c // опять чтобы не блочить каждое сложение
				}

				// для предшественника vPred
				// delta_s(vPred) += c_s(vPred, w)
				// дельта только для 's' и этой горутины, поэтому всё ок
				delta[vPredID] += c
			}
		}
		if len(sEdgeContributions) > 0 {
			mu.Lock()
			for edgeKey, val := range sEdgeContributions {
				cb[edgeKey] += val
			}
			mu.Unlock()
		}
	})
	return cb
}

// brandes_parallel - это общий распараллеленный код для BetweennessParallel и EdgeBetweennessParallel
// Соответствует algorithm 1 в http://algo.uni-konstanz.de/publications/b-vspbc-08.pdf
// Внешний цикл (по нодам 's') распараллелен (они могут считаться независимо).
func brandes_parallel(ctx context.Context, g graph.Graph, accumulate func(s graph.Node, stack linear.NodeStack, p map[int64][]graph.Node, delta, sigma map[int64]float64)) {
	nodesList := graph.NodesOf(g.Nodes())
	numGraphNodes := len(nodesList)

	if numGraphNodes == 0 {
		return
	}

	numWorkers := min(numGraphNodes, runtime.GOMAXPROCS(0))

	// Канал для передачи исходных узлов (s) воркерам.
	sChan := make(chan graph.Node, numGraphNodes)
	var wg sync.WaitGroup

	for range numWorkers {
		wg.Go(func() {
			// Структуры данных, локальные для каждого воркера.
			var workerStack linear.NodeStack

			// Initialize maps with a capacity hint based on the total number of nodes
			workerP := make(map[int64][]graph.Node, numGraphNodes) // список ID предшественников узла v на кратчайших путях от s.
			workerSigma := make(map[int64]float64, numGraphNodes)  // количество кратчайших путей от s до v.
			workerD := make(map[int64]int, numGraphNodes)          // расстояние (длина кратчайшего пути в ребрах) от s до v.
			workerDelta := make(map[int64]float64, numGraphNodes)  // вклад v в центральность других узлов из-за путей от s.
			var workerQueue linear.NodeQueue                       // Очередь для BFS.

			for s := range sChan {
				sID := s.ID()

				// Part 1 - BFS and Shortest Path Counting from s ---
				workerStack = workerStack[:0]

				for _, wNode := range nodesList {
					workerP[wNode.ID()] = workerP[wNode.ID()][:0]
				}

				// Инициализируем sigma (количество кратчайших путей) и d (расстояние) для всех узлов относительно s.
				for _, tNode := range nodesList {
					workerSigma[tNode.ID()] = 0
					workerD[tNode.ID()] = -1
				}
				workerSigma[sID] = 1
				workerD[sID] = 0

				workerQueue.Enqueue(s)
				for workerQueue.Len() != 0 {
					v := workerQueue.Dequeue()
					vID := v.ID()
					workerStack.Push(v) // BFS order

					// Итерируемся по соседям w узла v.
					iter := g.From(vID)
					for iter.Next() {
						w := iter.Node()
						wID := w.ID()
						if workerD[wID] < 0 {
							workerQueue.Enqueue(w)
							workerD[wID] = workerD[vID] + 1
						}
						if workerD[wID] == workerD[vID]+1 {
							workerSigma[wID] += workerSigma[vID]
							workerP[wID] = append(workerP[wID], v)
						}
					}
				}

				// Part 2 - Accumulation of Dependencies ---
				// Инициализируем delta (зависимости) для текущей s.
				for _, vNode := range nodesList {
					workerDelta[vNode.ID()] = 0
				}

				// При извлечении из стека, узлы (вершины) возвращаются в порядке невозрастающего расстояния от s.
				accumulate(s, workerStack, workerP, workerDelta, workerSigma)
			}
		})
	}
outerLoop:
	for _, sNode := range nodesList {
		select {
		case <-ctx.Done():
			break outerLoop
		case sChan <- sNode:
		}
	}
	close(sChan)

	wg.Wait()
}
