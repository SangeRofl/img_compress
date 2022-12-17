from PIL import Image
from random import random
import numpy as np
import time

def create_data(matrix, n, m):
    array = []
    blocks_in_line = IMAGE_WIDTH // m
    for i in range(IMAGE_HEIGHT // n):
        for y in range(n):
            line = []
            for j in range(blocks_in_line):
                for x in range(n):
                    pixel = []
                    for color in range(3):
                        pixel.append(trans_pix_1(matrix[i * blocks_in_line + j][(y * m * 3) + (x * 3) + color]))
                    line.append(tuple(pixel))
            array+=line
    return array






def create_data2(matrix, height, width):
    res = []
    
















def create_data1(matrix, n, m):
    array = []
    blocks_in_line = 64 // m
    for i in range(64 // n):
        for y in range(n):
            line = []
            for j in range(blocks_in_line):
                for x in range(n):
                    pixel = []
                    for color in range(3):
                        pixel.append(trans_pix_1(matrix[i * blocks_in_line + j][(y * m * 3) + (x * 3) + color]))
                    line.append(tuple(pixel))
            array+=line
    return array

def print_matrix(matrix):
    for i in matrix:
        for j in i:
            print(j, end = " ")
        print()

def trans_pix(val: int)->float:
    return ((2*val)/255)-1

def trans_pix_1(val: float)->int:
    return round(255*(val+1)/2)

def matrix_transp(matrix):
    if FAST_MODE:
        return np.matrix.transpose(np.array(matrix)).tolist()
    else:
        res = []
        for col in range(len(matrix[0])):
            column=[]
            for row in range(len(matrix)):
                column.append(matrix[row][col])
            res.append(column)
        return res

def matrix_diff(matrix1, matrix2):
    res = []
    for i in range(len(matrix1)):
        arr = []
        for j in range(len(matrix1[0])):
            arr.append(matrix1[i][j]-matrix2[i][j])
        res.append(arr)
    return res


def matrix_product(matrix1, matrix2):
    if FAST_MODE:
        m1=np.array(matrix1)
        m2=np.array(matrix2)
        return np.matmul(m1,m2).tolist()
    else:
        res = []
        for i in range(len(matrix1)):
            arr = []
            for j in range(len(matrix2[0])):
                sum = 0
                for k in range(len(matrix2)):
                    sum = sum + matrix1[i][k]*matrix2[k][j]
                arr.append(sum)
            res.append(arr)
        #print(res[0][0])
        return res


def scalar_product(a, matrix):
    res = []
    for i in range(len(matrix)):
        arr = []
        for j in range(len(matrix[0])):
            arr.append(matrix[i][j]*a)
        res.append(arr)
    return res

def split_image(image, square_width, square_height):
    im_w, im_h= image.size
    res = []#входные данные
    for y in range(0, im_h, square_width):
        for x in range(0, im_w, square_height):
            arr=[]
            for i in range(y, y+square_height):
                for j in range(x, x+square_width):
                    arr.append(trans_pix(pixels[j, i][0]))
                    arr.append(trans_pix(pixels[j, i][1]))
                    arr.append(trans_pix(pixels[j, i][2]))
            res.append(arr)
    return res

def work(input_data, W_1, W_2):
    res = []
    for x_i in input_data:
        x_i = [x_i]
        y_i = matrix_product(x_i, W_1)
        x_i_1 = matrix_product(y_i, W_2)
        res.append(x_i_1[0])
    return res


def study(input_data, neur_num, error):
    W_1 = []#матрица весов первого слоя
    for i in range(SQUARE_HEIGHT*SQUARE_WIDTH*3):
        arr = []
        for j in range(neur_num):
            arr.append(random()*2-1)
        W_1.append(arr)

    W_2 = matrix_transp(W_1)#матрица весов второго слоя
    E = 0
    while True:
        E = 0
        now = time.time()
        for x_i in input_data:
            
            x_i= [x_i]
            y_i = matrix_product(x_i, W_1)
            x_i_1 = matrix_product(y_i, W_2)
            d_x_i = matrix_diff(x_i_1, x_i)
            W_1 = matrix_diff(W_1, scalar_product(0.003, matrix_product(matrix_product(matrix_transp(x_i), d_x_i), matrix_transp(W_2))))
            W_2 = matrix_diff(W_2, scalar_product(0.003, matrix_product(matrix_transp(y_i), d_x_i)))
            #нормализация весовых коэффициентов
            if NORM_MODE:
                w1 = []
                for i in matrix_transp(W_1):
                    s = 0
                    for j in i:
                        s+=j**2
                    w1.append(s**0.5)

                for i in range(len(W_1)):
                    for j in range(len(W_1[0])):
                        # val = 0
                        # for k in range(len(W_1)):
                        #     val+=W_1[k][j]**2
                        # val = val**0.5
                        W_1[i][j] = W_1[i][j]/w1[j]
                w2 = []
                for i in matrix_transp(W_2):
                    s = 0
                    for j in i:
                        s+=j**2
                    w2.append(s**0.5)
                for j in range(len(W_2)):
                    for i in range(len(W_2[0])):
                        # val = 0
                        # for k in range(len(W_2)):
                        #     val+=W_2[k][i]**2
                        # val = val**0.5
                        W_2[j][i] = W_2[j][i]/w2[i]
            E_q = 0
            for i in d_x_i[0]:
                E_q+=i*i
            E+=E_q
        
        print("Время итерации: "+str(time.time()-now))    
        print("Ошибка: "+str(E))
        if E<error:
            break
    return W_1, W_2
def save_weights(W_1, W_2):
    f = open('weights.txt', 'wt')
    f.write(str(SQUARE_HEIGHT)+" "+str(SQUARE_WIDTH)+"\n")
    f.write(str(len(W_1))+" "+str(len(W_1[0]))+"\n")
    for i in range(len(W_1)):
        line = []
        for j in range(len(W_1[0])):
            line.append(str(W_1[i][j]))
        f.write(" ".join(line)+"\n")
        
    f.write(str(len(W_2))+" "+str(len(W_2[0]))+"\n")
    for i in range(len(W_2)):
        line = []
        for j in range(len(W_2[0])):
            line.append(str(W_2[i][j]))
        f.write(" ".join(line)+"\n")
    f.close()
def read_weights():
    f = open('weights.txt', 'rt')
    s = f.readline()[:-1].split(" ")
    SQUARE_WIDTH= int(s[1])
    SQUARE_HEIGHT = int(s[0])
    s = f.readline()[:-1].split(" ")
    w1_rows = int(s[0])
    w1_cols = int(s[1])
    W_1 = []
    for i in range(w1_rows):
        line = f.readline()[:-1].split(" ")
        for j in range(len(line)):
            line[j] = float(line[j])
        W_1.append(line)
    s = f.readline()[:-1].split(" ")
    w2_rows = int(s[0])
    w2_cols = int(s[1])
    W_2 = []
    for i in range(w2_rows):
        line = f.readline()[:-1].split(" ")
        for j in range(len(line)):
            line[j] = float(line[j])
        W_2.append(line)
    f.close()
    return W_1, W_2, SQUARE_HEIGHT, SQUARE_WIDTH

if __name__ == "__main__":
    im = Image.open('.\\data\\car1.png')
    pixels = im.load()
    IMAGE_WIDTH, IMAGE_HEIGHT= im.size
    print("Включить быстрый режим?(Y/n):")
    mode_dec=input().lower()
    FAST_MODE = (True if mode_dec=='y' or mode_dec=='д' or mode_dec=='1' else False)
    print("Режим работы/обучения?(1/0):")
    prog_mode = int(input())

    if(prog_mode == 0):
        print("Использовать нормализацию?(Y/n):")
        norm_dec=input().lower()
        NORM_MODE = (True if norm_dec=='y' or norm_dec=='д' or norm_dec=='1' else False)
        print("Введите размер прямоугольников:\nВысота: ")
        SQUARE_HEIGHT = int(input())#Высота
        print("Ширина: ")
        SQUARE_WIDTH = int(input())#Ширина
        X_0 = split_image(im, SQUARE_WIDTH, SQUARE_HEIGHT)
        print("Введите число нейронов второго слоя: ")
        p = int(input())
        print("Введите максимальную допустимую ошибку: ")
        e = float(input())
        print("Коэффициент сжатия: "+str(len(X_0)*SQUARE_WIDTH*SQUARE_HEIGHT*3/((len(X_0)+SQUARE_HEIGHT*SQUARE_WIDTH*3)*p+2)))
        W_1, W_2 = study(X_0, p, e)
        save_weights(W_1, W_2)
    else:
        W_1, W_2, SQUARE_HEIGHT, SQUARE_WIDTH = read_weights()
        im_2 = Image.new('RGB',(IMAGE_WIDTH,IMAGE_HEIGHT))
        X = work(split_image(im,SQUARE_WIDTH, SQUARE_HEIGHT), W_1, W_2)
        im_2.putdata(create_data(X, SQUARE_HEIGHT, SQUARE_WIDTH))
        im_2.save('.\\data\\output.jpg')




        
        

