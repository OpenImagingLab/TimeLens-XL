import time

class Ttracker():
	def __init__(self, track=False, count_step=10):
		self.if_track = track
		self.count = 0
		self.count_step = count_step
		self.record = {str(self.count):{}}
		self.last_time = time.time()
		self.count_start = None

	def count_plus(self):
		self.record[str(self.count)]['start'] = [time.time()-self.count_start]
		self.count += 1
		if self.count  % self.count_step == 0 and self.if_track:
			self.print()
		self.record.update({str(self.count):{}})

	def print(self):
		with open('time_analysis.txt', 'a+') as f:
			f.write(f'From {self.count-self.count_step} to {self.count-1}\n')
			keys = self.record[str(self.count-1)].keys()
			for k in keys:
				time_ana = 0.
				for count in self.record.keys():
					time_ana += self.record[count][k][0]
				f.write(f"{k}, avg: {time_ana/self.count_step*1000:.4f} ms\n")
			self.record = {}

	def time_init(self):
		self.last_time= time.time()

	def time_init_count(self):
		self.count_start = time.time()



	def track(self, record):
		if not self.if_track:
			return
		self.record[str(self.count)].update({
			str(record):[time.time()-self.last_time]
		})
		self.time_init()
		return
