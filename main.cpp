#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define pat 2
#define INPUT 2
#define HIDDEN 2
#define OUTPUT 1
#define min 1.0e-5
#define patternloop 4
#define ALL (INPUT+1)*HIDDEN+(HIDDEN+1)*OUTPUT

double OT_IN[INPUT + 1]; //入力層の出力-データセット入り
double OT_HN[HIDDEN + 1]; //中間層の出力(Hj)
double OT_OT[OUTPUT]; //出力層の出力(Ok)
double W_IN[pat][HIDDEN*(INPUT + 1)]; //結合係数(Wkj)
double W_HN[pat][OUTPUT*(HIDDEN + 1)]; //結合係数(Vkj)
double TEACH[OUTPUT]; //教師信号
double DEL_OT[OUTPUT]; //誤差(δk)
double DEL_HN[HIDDEN]; //誤差(σｊ)
double BFGS_H[ALL][ALL];
double BFGS_H2[ALL][ALL];
double BFGS_D[ALL];
double delta_E[pat][ALL];
double lx[ALL];

double ori_E;
double lx_E;
double alpha; //定数α
double erlimit; //誤差許容の上限値
double u0; //シグモイドの傾き
double intwgt; //結合係数の初期値
double intoff; //オフセットの初期値
double ru0; //変数
double error; //全体の誤差
double sum, wkb;
double tstart, tstop, ttime;
double atime = 0;

int gaku;
int times; //学習回数
int ok = 0;
int nook = 0;
int count;
int a, b, loop;
int sumloop = 0;
int flag = 0;
int loopf;

/*教師信号のデータセット*/
/*------------------*/
struct {
	double input[2];
	double tch[1];
} indata[4] = { { 1.0, 0.0, 0.99 }, { 0.0, 1.0, 0.99 }, { 0.0, 0.0, 0.01 }, { 1.0, 1.0, 0.01 }, };

/*乱数作成*/
/*--------*/
double drand48(){
	double r;
	r = (double)rand() / 32767;
	return r;
}

/*乱数作成*/
/*--------*/
double srand48(){
	double s = 0.0;
	while (s < 0.10 || s > 0.99) s = (double)rand() / 32767;
	return s;
}

/*シグモイド関数*/
/*-----------*/
double sigmoid(double u){
	double s;
	s = u / u0;
	s = 0.5*(1.0 + tanh(s));
	if (s>0.99000) return(0.99000);
	else if (s<0.0100) return(0.0100);
	else return(s);
}

/*データ入力*/
/*----------*/
void scan_data(FILE *fp){
	printf("シグモイドの傾き(u0)=");
	scanf_s("%lf", &u0);
	fprintf(fp, "シグモイドの傾き(u0)= %lf\n", u0);
	ru0 = 2.0 / u0;
	printf("誤差許容の上限値=");
	scanf_s("%lf", &erlimit);
	fprintf(fp, "誤差許容の上限値= %lf\n", erlimit);
	printf("学習回数の上限値=");
	scanf_s("%d", &times);
	fprintf(fp, "学習回数の上限値= %d\n", times);
}


/*データを乱数に初期化*/
/*------------------*/
void initial_data(FILE *fp){
	count = 0;
	int a, b;
	count = 0;
	for (a = 0; a<HIDDEN; a++){
		for (b = 0; b<INPUT; b++){
			W_IN[0][count] = intwgt*(drand48() - 0.5)*2.0;
			fprintf(fp,"W_IN[%d]=%f\n", count,W_IN[0][count]);
			count++;
		}
		W_IN[0][count] = intoff*drand48();
		fprintf(fp, "W_IN[%d]=%f\n", count, W_IN[0][count]);
		count++;
	}
	count = 0;
	for (a = 0; a<OUTPUT; a++){
		for (b = 0; b<HIDDEN; b++){
			W_HN[0][count] = intwgt*(drand48() - 0.5);
			fprintf(fp, "W_HN[%d]=%f\n", count, W_HN[0][count]);
			count++;
		}
		W_HN[0][count] = intoff*drand48();
		fprintf(fp, "W_HN[%d]=%f\n", count, W_HN[0][count]);
		count++;
	}
	for (a = 0; a < ALL; a++){
		for (b = 0; b < ALL; b++){
			if (a == b)
				BFGS_H[a][b] = 1.0;
			else
				BFGS_H[a][b] = 0.0;
		}
	}
}


/*バックプロバゲーション関数*/
/*------------------*/
void BP_INVERT(int pattern,FILE *fp){
	for (a = 0; a < ALL; a++)
		delta_E[pattern][a] = 0.0;
	for (loop = 0, error = 0.0, ori_E=0.0; loop < patternloop; loop++){
		if (loopf % 50 == 0) fprintf(fp, "Input  ");
		/*入力層の出力をセットする*/
		for (a = 0; a < INPUT; a++){
			OT_IN[a] = (double)indata[loop].input[a];
			if (loopf % 50 == 0) fprintf(fp, "%1.1f ", OT_IN[a]);
		}
		OT_IN[INPUT] = 1.0; //threshold/bias
		OT_HN[INPUT] = 1.0; //threshold/bias
		/*中間層の出力を求める*/
		count = 0;
		for (a = 0; a < HIDDEN; a++){
			for (b = 0, sum = 0.0; b < INPUT + 1; b++){
				sum += (W_IN[pattern][count] * OT_IN[b]);//Uj=Σ(i)Wji*Ii+θj
				count++;
			}
			OT_HN[a] = sigmoid(sum); //Hj=f(Uj)
		}
		/*出力層の出力を求める*/
		count = 0;
		for (a = 0; a < OUTPUT; a++){
			for (b = 0, sum = 0.0; b < HIDDEN + 1; b++){
				sum += (W_HN[pattern][count] * OT_HN[b]);//Sk=Σ(j)Vkj*Hj+γk
				count++;
			}
			OT_OT[a] = sigmoid(sum);//Ok=f(Sk)
			if (loopf % 50 == 0) fprintf(fp, "\nOutput  %lf\n", OT_OT[a]);
		}
		/*誤差の計算*/
		for (a = 0; a < OUTPUT; a++){
			TEACH[a] = (double)indata[loop].tch[a];
			wkb = TEACH[a] - OT_OT[a];
			ori_E += 0.5*(wkb*wkb);//E(x)の計算
			error += fabs(wkb);
			//δk=(Tk-Ok)*Ok*(1-Ok)
			DEL_OT[a] = wkb * ru0 * OT_OT[a] * (1.0 - OT_OT[a]);
		}
		/*誤差の計算*/
		count = 0;
		for (a = 0; a < HIDDEN; a++){
			for (b = 0,sum = 0.0; b < OUTPUT; b++){
				sum += (DEL_OT[b] * W_HN[pattern][count]);
				count++;
			}
			/*σj=Σ(k)δk*Vkj*Hj*(1-Hj)*/
			DEL_HN[a] = sum * ru0 * OT_HN[a] * (1.0 - OT_HN[a]);
		}

		/*ΔEの計算*/
		count = 0;
		for (a = 0; a < HIDDEN; a++){
			for (b = 0; b < INPUT + 1; b++){
				delta_E[pattern][count] += DEL_HN[a] * OT_IN[b];
				count++;
			}
		}
		for (a = 0; a < OUTPUT; a++){
			for (b = 0; b < HIDDEN + 1; b++){
				delta_E[pattern][count] +=  DEL_OT[a] * OT_HN[b];
				count++;
			}
		}
	}
}

void line_x(){
	for (a = 0, count = 0; a < (INPUT + 1)*HIDDEN; a++){
		lx[count] = W_IN[0][a] + alpha * BFGS_D[count];
		count++;
	}
	for (a = 0; a < (HIDDEN+1)*OUTPUT;a++){
		lx[count] = W_HN[0][a] + alpha * BFGS_D[count];
		count++;
	}
	for (loop = 0, lx_E = 0.0; loop < patternloop; loop++){
		/*入力層の出力をセットする*/
		for (a = 0; a < INPUT; a++){
			OT_IN[a] = (double)indata[loop].input[a];
		}
		OT_IN[INPUT] = 1.0; //threshold/bias
		/*中間層の出力を求める*/
		count = 0;
		for (a = 0; a < HIDDEN; a++){
			for (b = 0, sum = 0.0; b < INPUT + 1; b++){
				sum += (lx[count] * OT_IN[b]);//Uj=Σ(i)Wji*Ii+θj
				count++;
			}
			OT_HN[a] = sigmoid(sum); //Hj=f(Uj)
		}
		OT_HN[INPUT] = 1.0; //threshold/bias
		/*出力層の出力を求める*/
		for (a = 0; a < OUTPUT; a++){
			for (b = 0, sum = 0.0; b < HIDDEN + 1; b++){
				sum += (lx[count] * OT_HN[b]);//Sk=Σ(j)Vkj*Hj+γk
				count++;
			}
			OT_OT[a] = sigmoid(sum);//Ok=f(Sk)
		}
		for (a = 0; a < OUTPUT; a++){
			TEACH[a] = (double)indata[loop].tch[a];
		}
		for (a = 0; a < OUTPUT;a++)
			lx_E += 0.5*((TEACH[a] - OT_OT[a])*(TEACH[a] - OT_OT[a]));
	}
}

void BFGS_DK(){
	for (a = 0; a<ALL; a++){
		BFGS_D[a] = 0.0;
		for (b = 0; b<ALL; b++){
			BFGS_D[a] += -(BFGS_H[a][b]) * delta_E[0][b];
		}
	}
}

void BFGS_ALPHA(){
	//Armijo Rule
	double FTD;
	FTD = 0.0;
	alpha = 1.0;
	for (a = 0; a < ALL; a++){
		FTD += delta_E[0][a] * BFGS_D[a];
	}
	line_x();
	for (int dim=0; dim<10; dim++){
		line_x();
		if (lx_E <= ori_E + 0.01* alpha * FTD){
			break;
		}
		else	
			alpha *= 0.5;
	}
}

void BFGS_XUPDATE(){
	int count_WHN = 0;
	int count_WIN = 0;
	for (a = 0; a<ALL; a++){
		if (a < INPUT*(HIDDEN + 1)){
			W_IN[1][count_WIN] = W_IN[0][count_WIN] + alpha * BFGS_D[a];
			count_WIN++;
		}
		else {
			W_HN[1][count_WHN] = W_HN[0][count_WHN] + alpha * BFGS_D[a];
			count_WHN++;
		}
	}
}

void BFGS_flagx(){
	int done = 0;
	for (a = 0; a<ALL; a++){
		if (fabs(delta_E[0][a]) <= min){
			done++;
		}
	}
	if (done == ALL)
		flag = 1;
}

void BFGS_HK(){
	double y[ALL], s[ALL], sTy, Hy[ALL], HysT[ALL][ALL], sHyT[ALL][ALL], yTHy, ssT[ALL][ALL];
	int i = 0, j = 0;
	sTy = 0.0;
	yTHy = 0.0;
	for (a = 0; a<ALL; a++){
		Hy[a] = 0.0;
		y[a] = delta_E[1][a] - delta_E[0][a]; //yk=∇f(x+1)-∇f(x)
		if (a < INPUT*(HIDDEN + 1)){
			s[a] = W_IN[1][i] - W_IN[0][i];
			i++;
		}
		else{
			s[a] = W_HN[1][j] - W_HN[0][j];
			j++;
		}
	}
	for (a = 0; a<ALL; a++){
		sTy += s[a] * y[a];
		for (b = 0; b<ALL; b++){
			Hy[a] += BFGS_H[a][b] * y[b];
			ssT[a][b] = s[a] * s[b];
		}
	}
	for (a = 0; a<ALL; a++){
		yTHy += y[a] * Hy[a];
		for (b = 0; b<ALL; b++){
			HysT[a][b] = Hy[a] * s[b];
			sHyT[a][b] = s[a] * Hy[b];
		}
	}
	for (a = 0; a<ALL; a++){
		for (b = 0; b<ALL; b++){
			BFGS_H2[a][b] = BFGS_H[a][b] - ((HysT[a][b] + sHyT[a][b]) / sTy) + (1 + (yTHy / sTy))*(ssT[a][b] / sTy);
		}
	}

	//HKとｘ更新
	for (a = 0; a<ALL; a++){
		delta_E[0][a] = delta_E[1][a];
		for (b = 0; b<ALL; b++)
			BFGS_H[a][b] = BFGS_H2[a][b];
	}
	for (a = 0; a<(INPUT + 1)*HIDDEN; a++){
		W_IN[0][a] = W_IN[1][a];
	}
	for (a = 0; a<(HIDDEN + 1)*OUTPUT; a++){
		W_HN[0][a] = W_HN[1][a];
	}
}


void BFGS_BP(FILE *fp){
	loopf = 0;
	BP_INVERT(0, fp);
	tstart = (double)clock();
	while (loopf < times){
		loopf++;
		if (loopf % 50 == 0) fprintf(fp, "\n学習第%d回\n", loopf);
		BFGS_DK();
		BFGS_ALPHA();
		BFGS_XUPDATE();
		BP_INVERT(1, fp);
		BFGS_HK();


		/****(ORIGINAL BP)testing purpose only*****/

		/*BP_INVERT(0, fp);
		alpha = 0.8;
		count = 0;
		for (a = 0; a < INPUT*(HIDDEN + 1); a++){
			W_IN[0][a] += alpha * delta_E[0][count];
			count++;
		}
		for (a = 0; a < OUTPUT*(HIDDEN + 1); a++){
			W_HN[0][a] += alpha * delta_E[0][count];
			count++;
		}*/

		/****************************/

		if (loopf % 50 == 0) fprintf(fp, "-----------\nError  %lf\n", error);
		if (error <= erlimit){
			ok++;
			sumloop += loopf;
			tstop = (double)clock();
			ttime = tstop - tstart;
			atime += ttime;
			break;
		}
		if (loopf == times && error > erlimit){
			tstop = (double)clock();
			nook++;
		}
	}
	fprintf(fp, "\n-----------------\n");
	fprintf(fp, "全体の学習回数%d\n", loopf);
}


void print_w(FILE *fp){
	count = 0;
	for (a = 0; a<HIDDEN; a++){
		fprintf(fp, "結合係数W[%d] = {", a);
		for (b = 0; b<INPUT; b++){
			if (b != 0) fprintf(fp, ",");
			fprintf(fp, "%.6lf", W_IN[0][count]);
			count++;
		}
		fprintf(fp, "}  シータ = %.6lf\n", W_IN[0][count]);
		count++;
	}
	count = 0;
	for (a = 0; a<OUTPUT; a++){
		fprintf(fp, "結合係数V[%d] = {", a);
		for (b = 0; b<HIDDEN; b++){
			if (b != 0) fprintf(fp, ",");
			fprintf(fp, "%.6lf", W_HN[0][count]);
			count++;
		}
		fprintf(fp, "}  ガンマ = %.6lf\n", W_HN[0][count]);
		count++;
	}
}

int main(void){
	double b;
	FILE *fp;
	fopen_s(&fp, "out.txt", "w+");
	scan_data(fp);
	
	for (gaku = 0; gaku < 10; gaku++){
		srand(gaku);
		intwgt = srand48();
		intoff = srand48();
		fprintf(fp, "結合係数の初期値[-1.0,1.0] * ? = %lf\n", intwgt);
		fprintf(fp, "オフセットの初期値[0.0,1.0] * +-? = %lf\n", intoff);
		fprintf(fp, "\n----データセット----\n");
		fprintf(fp, "1 1->(0)\n0 1->(1)\n0 0->(0)\n1 1->(0)\n");
		initial_data(fp);
		BFGS_BP(fp);
		print_w(fp);
		fprintf(fp, "成功=%d,失敗=%d\n", ok, nook);
		if (ok != 0){
			(double)b = sumloop / ok;
			fprintf(fp, "反復平均回数=%lf\n", b);
		}
		printf("done(%d)\n", gaku);
	}
	atime = atime / (double)ok;
	fprintf(fp, "平均時間=%lf", atime);
	fclose(fp);
	return 0;
}



