from learning_union_ktss_master.DFA import DFA
import copy
import uuid

class k_testable():
	"""
		Class for Deterministic finite automaton (DFA)
	"""

	def __init__(self, E=[], I={}, F={}, T={}, C={}):
		"""
			Constructor for k_testable class. Returns a k_testable
				E : e
				I : e
				F : e
				T : e
				C : e
		"""
		self.E = E
		self.I = I
		self.F = F
		self.T = T
		self.C = C

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			if sorted(self.E) == sorted(other.E):
				if sorted(self.I) == sorted(other.I):
					if sorted(self.F) == sorted(other.F):
						if sorted(self.T) == sorted(other.T):
							if sorted(self.C) == sorted(other.C):
								return True
		return False

	def __hash__(self):
		s = str(sorted(self.E)) + str(sorted(self.I)) + str(sorted(self.F)) + str(sorted(self.T)) + str(sorted(self.C))
		return hash(s)

	def __str__(self):
		return str(sorted(self.E)) + str(sorted(self.I)) + str(sorted(self.F)) + str(sorted(self.T)) + str(sorted(self.C))


def fusion_of_kss(k1,k2):
	fusion = copy.deepcopy(k1)
	fusion.E = list(set(fusion.E + list(set(k2.E))))
	fusion.I.update(k2.I)
	fusion.F.update(k2.F)
	fusion.T.update(k2.T)
	fusion.C.update(k2.C)
	return fusion


def calculateEIFTC(X,k):
	# Σ(X) is the alphabet used in X
	sigma = []
    
	for x in X:
		sigma = list(set(sigma + list(x.split(' '))))
        
	kminus1 = k-1
	
	# I(S)= Σ(S)k-1 ∩ Pref(S)
	I = {}
	for x in X:
		xs = x.split(' ')
		if len(xs)>=k-1:
			I[" ".join(xs[:kminus1])] = None

	# F(S)= Σ(S)k-1 ∩ Suff(S)
	F = {}
	if k > 1:
		for x in X:
			xs = x.split(' ')
			if len(xs)>=k-1:
				F[" ".join(xs[-kminus1:])] = None

	# T(S)= Σ(S)k ∩ {v: uvw in S}
	T = {}
    
    #*** Weighthed
	W = {}
#    count = 1
	for x in X:
		xs = x.split(' ')

		if len(xs)<k:
			continue
		start = 0
		end   = k
		while end <= len(xs):
			segment = " ".join(xs[start:end])
            
			if segment not in T:
				T[segment] = 1
			else:
				T[segment] +=1                
			start += 1
			end   += 1
            
            #*** Weighthed
			if segment not in W:
				W[segment]=1
			else :
				W[segment]+=1
                
                
	#C(S)= Σ(S)<k ∩ S
	C = {}
	for x in X:
		xs = x.split(' ')

        	
		if len(xs)<k:
	#		if len(xs)>2:
			C[" ".join(xs)] = None

	return sigma,I,F,T,C,W



def has_crossovers(E,I,F,T,I_prime,F_prime,T_prime):
	#condition1
	diff_I       = I.keys() - I_prime
	diff_T_prime = T_prime.keys() - T
	for w in diff_I:
		for a in E:
			if w+a in diff_T_prime:
				# print('cond1',w,w+a)
				return (w,w+a)
	#condition2
	diff_T       = T.keys() - T_prime
	for w in diff_T:
		a = w[:1]
		b = w[1:]
		for w_prime in diff_T_prime:
			c = w_prime[:-1]
			d = w_prime[-1:]
			if b==c:
				# print('cond2',w,w_prime)
				return (w,w_prime)
	#condition3
	diff_F       = F.keys() - F_prime
	for w in diff_F:
		for a in E:
			if a+w in diff_T_prime:
				# print('cond3',w,a+w)
				return (w,a+w)

	return None




def check_union(alphabet,I,F,T,I_prime,F_prime,T_prime,k):
	"""
		Checks whether \gamma_k (Z \sqcup Z') = \gamma_k(Z) \cup \gamma_k(Z') holds
	"""
	if I.keys() <= I_prime.keys() and F.keys() <= F_prime.keys() and T.keys() <= T_prime.keys():
		return True
	if I_prime.keys() <= I.keys() and F_prime.keys() <= F.keys() and T_prime.keys() <= T.keys():
		return True
	
	#Compute vertices s.t. V = \{ \bullet u \mid u \in I \cup I' \} \cup T \cup T' \cup \{ u \bullet \mid u \in F \cup F' \}
	FRESH_SYMBOL = '#'
	alphabet     = {symbol:None for symbol in alphabet}
	automaton    = DFA("", {}, {}, {}, {}, alphabet=alphabet)
	red   = []
	blue  = []
	white = []
	for t in T:
		automaton.states.update({t:None})
		if t in (T.keys() - T_prime.keys()):
			red.append(t)
		elif t in (T_prime.keys() - T.keys()):
			blue.append(t)
		elif t in (T.keys() & T_prime.keys()):
			white.append(t)
	for t in T_prime:
		automaton.states.update({t:None})
		if t in (T.keys() - T_prime.keys()):
			red.append(t)
		elif t in (T_prime.keys() - T.keys()):
			blue.append(t)
		elif t in (T.keys() & T_prime.keys()):
			white.append(t)
	for f in F:
		automaton.states.update({f+FRESH_SYMBOL:None})
		if f in (F.keys() - F_prime.keys()):
			red.append(f+FRESH_SYMBOL)
		elif f in (F_prime.keys() - F.keys()):
			blue.append(f+FRESH_SYMBOL)
		elif f in (F.keys() & F_prime.keys()):
			white.append(f+FRESH_SYMBOL)
	for f in F_prime:
		automaton.states.update({f+FRESH_SYMBOL:None})
		if f in (F.keys() - F_prime.keys()):
			red.append(f+FRESH_SYMBOL)
		elif f in (F_prime.keys() - F.keys()):
			blue.append(f+FRESH_SYMBOL)
		elif f in (F.keys() & F_prime.keys()):
			white.append(f+FRESH_SYMBOL)
	for i in I:
		automaton.states.update({FRESH_SYMBOL+i:None})
		if i in (I.keys() - I_prime.keys()):
			red.append(FRESH_SYMBOL+i)
		elif i in (I_prime.keys() - I.keys()):
			blue.append(FRESH_SYMBOL+i)
		elif i in (I.keys() & I_prime.keys()):
			white.append(FRESH_SYMBOL+i)
	for i in I_prime:
		automaton.states.update({FRESH_SYMBOL+i:None})
		if i in (I.keys() - I_prime.keys()):
			red.append(FRESH_SYMBOL+i)
		elif i in (I_prime.keys() - I.keys()):
			blue.append(FRESH_SYMBOL+i)
		elif i in (I.keys() & I_prime.keys()):
			white.append(FRESH_SYMBOL+i)
	#Compute edges s.t. E = \{ (au, ub) \in V \times V \mid a, b \in \Sigma \cup \{ \bullet \}, u \in \Sigma^{k-1} \}
	ID_TRANS = 1
	for au in automaton.states:
		for ub in automaton.states:
			if au[1:] == ub[:-1]:
				try:
					automaton.transitions[au][ID_TRANS] = ub
					ID_TRANS += 1
				except KeyError:
					automaton.transitions[au] = {}
					automaton.transitions[au][ID_TRANS] = ub
					ID_TRANS += 1
	#$\gamma_k (Z \sqcup Z') = \gamma_k(Z) \cup \gamma_k(Z')$ iff there exists a path in $G$ from a red vertex to a blue vertex, or from a blue vertex to a red vertex.
	for r in red:
		for b in blue:
			if automaton.is_reachable_from(r,b):
				return False
			if automaton.is_reachable_from(b,r):
				return False
	return True


def learnKtestable_fromEIFTC(E,I,F,T,C,k):
	#Create an automaton
	alphabet = {symbol:None for symbol in E}
	kss = DFA("", {}, {}, {}, {}, alphabet=alphabet)

	#Each string in I U C and Pref(I U C) is a state 
	for state in I:
		state = state.split(" ")
		kss.states["".join(state)] = None
		for i in range(1,len(state)+1):
			kss.states["".join(state[:i])] = None
			try:
				kss.transitions["".join(state[:i-1])]["".join(state[:i][-1:])] = "".join(state[:i])
			except KeyError:
				kss.transitions["".join(state[:i-1])] = {}
				kss.transitions["".join(state[:i-1])]["".join(state[:i][-1:])] = "".join(state[:i])
	#print(kss.toDot())
	for state in C:
		state = state.split(" ")

		kss.states["".join(state)] = None
		for i in range(1,len(state)+1):
			kss.states["".join(state[:i])] = None
			try:
				kss.transitions["".join(state[:i-1])]["".join(state[:i][-1:])] = "".join(state[:i])
			except KeyError:
				kss.transitions["".join(state[:i-1])] = {}
				kss.transitions["".join(state[:i-1])]["".join(state[:i][-1:])] = "".join(state[:i])

	# Each substring of length k-1 of strings in T is a state;
	for x in T:
		x = x.split(" ")

		if len(x)<k:
			continue
		start = 0
		end   = k-1
		while end <= len(x):
			kss.states["".join(x[start:end])] = None
			start += 1
			end   += 1
			
	# λ is the initial state;
	kss.states[""] = None

	# Add a transition labeled b from au to ub for each aub in T;
	for x in T:
		x = x.split(" ")
		a = "".join(x[:1])
		u = "".join(x[1:-1])
		b = "".join(x[-1:])
		try:
			kss.transitions[a+u][b] = u+b
		except KeyError:
			kss.transitions[a+u] = {}
			kss.transitions[a+u][b] = u+b

	# Each state/substring that is in F is a final state.
	for x in F:
		x = x.split(" ")
		newx = "".join(x)        
		if newx in kss.states:
			kss.acceptingStates[newx] = None

	# Each state/substring that is in C is a final state.
	for x in C:
		x = x.split(" ")
		newx = "".join(x)
		if newx in kss.states:
			kss.acceptingStates[newx] = None
	kss.clean()
	return kss