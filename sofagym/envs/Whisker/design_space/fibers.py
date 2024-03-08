import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import math as m
# import pylustrator
# activate pylustrator
# pylustrator.start()
def fibers(number_chambers = 2,r = 11, p = 2, H = 142.3862887, cutting_plane = None,
		coarse_points = 3000, fine_points = 80000, h = 25, draw = False):
	final_helix = []
	for i in range(number_chambers):
		final_helix.append([])

	def init_helix_path(r = r, p = p, H = H, number_points=None, h = h, draw = False, save = False):
		# r: bán kính của đường xoắn ốc ở mặt phẳng đáy
		# h: chiều cao của đường xoắn ốc cụt (chamber)
		# H: chiều cao của đường xoắn ốc (khoảng cách từ điểm đầu đến điểm cuối) 
		# number_points: số lượng các node hình thành nên đường cong (init: coarse)
		# p: thread
		idx_z_limit = int(h*number_points/H)
		# Tính toán tọa độ của các điểm trên đường xoắn ốc
		z = np.linspace(0, H, number_points)
		x = ((H-z)/H)*r*np.sin((2*np.pi)*z/p+np.pi)
		y = ((H-z)/H)*r*np.cos((2*np.pi)*z/p+np.pi)
		init_curve1 = [[x[i], y[i], z[i]] for i in range(number_points)]

		z2 = np.linspace(0, H, number_points)
		x2 = -((H-z2)/H)*r*np.sin(-(2*np.pi)*z2/p-np.pi)
		y2 = -((H-z2)/H)*r*np.cos(-(2*np.pi)*z2/p-np.pi)
		init_curve2 = [[x2[i], y2[i], z2[i]] for i in range(number_points)]
		init_curve = [init_curve1,init_curve2]
		if draw:
			### Draw Figures
			# Vẽ đường xoắn ốc
			fig = plt.figure()
			ax = fig.add_subplot(111,projection='3d')
			ax.plot(x, y, z, label='fiber 1')
			ax.plot(x2, y2, z2, label='fiber 2')
			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.set_zlabel('Z')
			ax.set_zlim([0, h])
			ax.legend()
		if save:
			for i in range(2):
				filePath_node = os.getcwd()+"/fiber"+str(i+1)+"_info.csv"
				try:
					os.remove(filePath_node)
				except:
					print("Error while deleting file ", filePath_node)
				header = ["Fiber "+ str(i+1)+": Spring Idx", "x (mm)", "y (mm)", "z (mm)"]
				with open(filePath_node, "w", newline="") as file_open:
					writer = csv.writer(file_open, delimiter=",")
					writer.writerow(header)
					for j in range(idx_z_limit):
						if i == 0 and init_curve[i][j][2]<h:
							writer.writerow([j,x[j],y[j],z[j]])
						elif i == 1 and init_curve[i][j][2]<h:
							writer.writerow([j,x2[j],y2[j],z2[j]])
		
		return init_curve

	def trimming_helix(number_points = coarse_points, cutting_plane = cutting_plane, draw = False):
		# cutting_plane  # Plane parallel to xoz plane
		# Initialize lists to store trimmed points
		
		init_fine_curve = init_helix_path(number_points = number_points)

		# Trim the first curve by the yoz plane
		trimmed_points_curve = []
		trimming = []
		for i in range(len(init_fine_curve)):
			trimming.append([])
			for j in range(number_points):
				if init_fine_curve[i][j][0] >= cutting_plane and cutting_plane>0:
					trimming[-1].append(init_fine_curve[i][j])
				if init_fine_curve[i][j][0] <= cutting_plane and cutting_plane<0:
					trimming[-1].append(init_fine_curve[i][j])
		if len(trimming[0]) - len(trimming[1]) > 0:
			trimmed_points_curve = (trimming[0][:-(len(trimming[0]) - len(trimming[1]))], trimming[1])
		elif len(trimming[0]) - len(trimming[1]) < 0:
			trimmed_points_curve = (trimming[0], trimming[1][:len(trimming[0]) - len(trimming[1])])
		else:
			trimmed_points_curve = trimming

		trimmed_points_curve= np.array(trimmed_points_curve)

		
		if draw:
			### Draw Figures
			# Vẽ đường xoắn ốc
			fig = plt.figure()
			ax = fig.add_subplot(111,projection='3d')
			ax.plot(trimmed_points_curve[0][:,0], trimmed_points_curve[0][:,1], trimmed_points_curve[0][:,2], label='fiber 1')
			# ax.plot(trimmed_points_curve[1][:,0], trimmed_points_curve[1][:,1], trimmed_points_curve[1][:,2], label='fiber 2')
			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.set_zlabel('Z')
			ax.set_zlim([0, h])
			ax.legend()
		return trimmed_points_curve

	def sort_trimmed_curves(trimmed_points_curve,cutting_plane = cutting_plane):
		point_in_plane = []
		for i in range(2):	
			point_in_plane.append([])
			for j in range(int(h/2)):
				if cutting_plane > 0:
					if i == 0:
						max_y = [max(trimmed_points_curve[i][:, 1]),min(trimmed_points_curve[i][:, 2])]
						min_y = [min(trimmed_points_curve[i][:, 1])]
						for sorted_indices in range(len(trimmed_points_curve[i])):
							if trimmed_points_curve[i][sorted_indices][1] == min_y[0]:
								min_y.append(trimmed_points_curve[i][sorted_indices][2])
						point_in_plane[i].append([cutting_plane, max_y[0]-(2*j)/m.tan(85.5*m.pi/180), max_y[1]+2*j])
						point_in_plane[i].append([cutting_plane, min_y[0]+(2*j)/m.tan(85.5*m.pi/180), min_y[1]+2*j])
					if i == 1:
						max_y = [min(trimmed_points_curve[i][:, 1]),min(trimmed_points_curve[i][:, 2])]
						min_y = [max(trimmed_points_curve[i][:, 1])]
						for sorted_indices in range(len(trimmed_points_curve[i])):
							if trimmed_points_curve[i][sorted_indices][1] == min_y[0]:
								min_y.append(trimmed_points_curve[i][sorted_indices][2])
						point_in_plane[i].append([cutting_plane, max_y[0]+(2*j)/m.tan(85.5*m.pi/180), max_y[1]+2*j])
						point_in_plane[i].append([cutting_plane, min_y[0]-(2*j)/m.tan(85.5*m.pi/180), min_y[1]+2*j])	
				if cutting_plane < 0:
					if i == 0:
							max_y = [min(trimmed_points_curve[i][:, 1]),min(trimmed_points_curve[i][:, 2])]
							min_y = [max(trimmed_points_curve[i][:, 1])]
							for sorted_indices in range(len(trimmed_points_curve[i])):
								if trimmed_points_curve[i][sorted_indices][1] == min_y[0]:
									min_y.append(trimmed_points_curve[i][sorted_indices][2])
							point_in_plane[i].append([cutting_plane, max_y[0]+(2*j)/m.tan(85.5*m.pi/180), max_y[1]+2*j])
							point_in_plane[i].append([cutting_plane, min_y[0]-(2*j)/m.tan(85.5*m.pi/180), min_y[1]+2*j])
					if i == 1:
						max_y = [max(trimmed_points_curve[i][:, 1]),min(trimmed_points_curve[i][:, 2])]
						min_y = [min(trimmed_points_curve[i][:, 1])]
						for sorted_indices in range(len(trimmed_points_curve[i])):
							if trimmed_points_curve[i][sorted_indices][1] == min_y[0]:
								min_y.append(trimmed_points_curve[i][sorted_indices][2])
						point_in_plane[i].append([cutting_plane, max_y[0]-(2*j)/m.tan(85.5*m.pi/180), max_y[1]+2*j])
						point_in_plane[i].append([cutting_plane, min_y[0]+(2*j)/m.tan(85.5*m.pi/180), min_y[1]+2*j])			
		point_in_plane = np.array(point_in_plane)
		return point_in_plane

	def save_csv(final_helix, cutting_plane = cutting_plane):
		# Save node information of original curves
		if cutting_plane > 0:
			name = "right"
		else:
			name = "left"
		for i in range(2):
			filePath_node = os.getcwd()+"/fiber"+str(i+1)+name+"_info.csv"
			try:
				os.remove(filePath_node)
			except:
				print("Error while deleting file ", filePath_node)
			header = ["Fiber "+ str(i+1)+"_"+name+": Idx", "x (mm)", "y (mm)", "z (mm)", "Cutting plane x"]
			with open(filePath_node, "w", newline="") as file_open:
				writer = csv.writer(file_open, delimiter=",")
				writer.writerow(header)
				data_limit = int((h/2)*coarse_points/H)+len(point_in_plane[i])
				for j in range(len(final_helix[i])):
					if final_helix[i][j][2]<h:
						writer.writerow([j,final_helix[i][j][0],final_helix[i][j][1],final_helix[i][j][2],cutting_plane])
	
	init_helix_path(number_points= coarse_points, draw=0, save=0)
	init_curve = trimming_helix(number_points=coarse_points, draw=0)
	trimmed_points_curve = trimming_helix(number_points=fine_points, draw=0)
	point_in_plane = sort_trimmed_curves(trimmed_points_curve=trimmed_points_curve)
	final_helix = []
	for i in range(len(init_curve)):
		revised_helix = init_curve[i]
		final_helix.append([])
		data_limit = int((h/2)*coarse_points/H)+len(point_in_plane[i])
		for j in range(0,len(point_in_plane[i])):
			# Find the index to insert the new point
			insert_index_z = np.searchsorted(revised_helix[:, 2], point_in_plane[i][j][2],side='right')
			# fix odd points
			if abs(revised_helix[insert_index_z][2] - revised_helix[insert_index_z-1][2]) < 0.3:
				insert_index_z -= 1
			# Insert the new point at the calculated index
			revised_helix = np.insert(revised_helix, insert_index_z, point_in_plane[i][j], axis=0)	
		for k in range(0,int(len(point_in_plane[i])/2)-1):
			n = 12  # Change this to the desired number of points
			step_size = 1.0 / (n - 1)
			intermediate_points = []
			for g in range(n):
				interpolated_point = (1 - g * step_size) * point_in_plane[i][2*k+1] + g * step_size * point_in_plane[i][2*k+2]
				intermediate_points.append(interpolated_point)
			for j in range(len(revised_helix)-1):
				if np.array_equal(revised_helix[j], intermediate_points[0]):
					revised_helix = np.insert(revised_helix, j+1, intermediate_points[1:n-1], axis=0)
					break
		for j in range(len(revised_helix)):
			if revised_helix[j][2]<h:
				final_helix[-1].append(revised_helix[j])

	final_helix = np.array(final_helix)
	# save csv
	save_csv(final_helix=final_helix)
	if draw:
	### Draw Figures
	# Vẽ đường xoắn ốc 
		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
		no = 400
		ax.plot(final_helix[0][0:no,0], final_helix[0][0:no,1], final_helix[0][0:no,2], marker='o', label='fiber 1')
		ax.plot(final_helix[1][0:no,0], final_helix[1][0:no,1], final_helix[1][0:no,2], marker='*', label='fiber 2')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.set_zlim([0, h])
		ax.legend()    
		plt.show()
	return final_helix

# Function used only if this script is called from a python environment
if __name__ == '__main__':	
	final_helix = fibers(cutting_plane = 1.1, draw=1)
	# final_helix = fibers(cutting_plane = -1, draw=1)
