all:
	cd V0_Hybride && make
	cd V0_Open_MP && make
	cd V0_MPI_pur && make
	cd V1_MPI_pur && make
	cd V1_Open_MP && make
	cd V1_hybride && make
	cd V1_Améliorée_MPI_pur && make
	cd V1_amélioré_Hybride && make
	cd V2_MPI_pur-gauche && make
	cd V2_MPI_PUR-droite && make
	cd V2_Open_MP && make

clean:
	cd V0_Hybride && make clean
	cd V0_Open_MP && make clean
	cd V0_MPI_pur && make clean
	cd V1_MPI_pur && make clean
	cd V1_Open_MP && make clean
	cd V1_hybride && make clean
	cd V1_Améliorée_MPI_pur && make clean
	cd V1_amélioré_Hybride && make clean
	cd V2_MPI_pur-gauche && make clean
	cd V2_MPI_PUR-droite && make clean
	cd V2_Open_MP && make clean

