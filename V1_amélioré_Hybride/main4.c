#include "projet.h"
#include <sys/time.h>
/* 2017-02-23 : version 1.0 améliorée hybride */
#include <mpi.h>
#include <omp.h>
#define P 5 //profondeur >PROF_PARAL jusqu'à laquelle 0 évalue les coups descendants du dernier coup avant de les distribuer
#define PROF_PARAL 7 //profondeur à partir de laquelle le programme se fait en parallèle
#define TAG 10 //dans ce programme, toutes les communications MPI se font avec ce tag

double my_gettimeofday(){
	struct timeval tmp_time;
	gettimeofday(&tmp_time, NULL);
	return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

unsigned long long int node_searched = 0;

void evaluate(tree_t * T, result_t *result, int my_rank,MPI_Status* status,int NP)
{
	node_searched++;
  
        move_t moves[MAX_MOVES];
        int n_moves;

        result->score = -MAX_SCORE - 1;
        result->pv_length = 0;
        
        if (test_draw_or_victory(T, result))
          return;

        if (TRANSPOSITION_TABLE && tt_lookup(T, result))     /* la réponse est-elle déjà connue ? */
          return;
        
        compute_attack_squares(T);

        /* profondeur max atteinte ? si oui, évaluation heuristique */
        if (T->depth == 0) {
		result->score = (2 * T->side - 1) * heuristic_evaluation(T);
		return;
        }
        
        n_moves = generate_legal_moves(T, &moves[0]);

        /* absence de coups légaux : pat ou mat */
	//printf("nmoves = %d\n",n_moves);
	if (n_moves == 0) {
		result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
		return;
        }
        
        if (ALPHA_BETA_PRUNING)
		sort_moves(T, n_moves, moves);



		//À PARALLELISER, le maître distribue le travail (un coup à chacun)
	if(my_rank==0){
		/* Création de la structure result pour MPI */
		const int nbChampsResult=4;
		int tailleChampsResult[4] = {1,1,1, MAX_DEPTH};
		MPI_Datatype TypesChampsResult[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
		MPI_Datatype MPI_RESULT_T;
		MPI_Aint     StructResult[4];

		StructResult[0] = offsetof(result_t, score);
		StructResult[1] = offsetof(result_t, best_move);
		StructResult[2] = offsetof(result_t, pv_length);
		StructResult[3] = offsetof(result_t, PV);

		MPI_Type_create_struct(nbChampsResult, tailleChampsResult, StructResult, TypesChampsResult, &MPI_RESULT_T);
		MPI_Type_commit(&MPI_RESULT_T);

		int i;       
		int continuer;
		int a;
		result_t child_result;

		/*on va stocker les résultats dans un tableau*/
		result_t* tableau=(result_t*)malloc(sizeof(result_t)*n_moves);
		MPI_Request* request=(MPI_Request*)malloc(sizeof(MPI_Request)*n_moves);

		/*Première profondeur*/
		if(T->height==0){

			/*Distribution des n_moves-1 coups*/
			for (i = 0; i < n_moves-1; i++) {
				continuer=1;
				MPI_Recv(&a, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, status);
				MPI_Send(&continuer, 1, MPI_INT, status->MPI_SOURCE, TAG, MPI_COMM_WORLD);
				MPI_Send(&(T->depth), 1, MPI_INT, status->MPI_SOURCE, TAG, MPI_COMM_WORLD);
				MPI_Send(&(moves[i]), 1, MPI_INT, status->MPI_SOURCE, TAG, MPI_COMM_WORLD);

				/*On recoit en non bloquant les résultats des fils que l'on envoie*/
				MPI_Irecv(tableau+i,1,MPI_RESULT_T,status->MPI_SOURCE,TAG,MPI_COMM_WORLD,request+i);
			}
			
			tree_t child;

			/*On joue le dernier coup*/
			play_move(T, moves[n_moves-1], &child);

			/*Le processus 0 descend récursivement à droite de l'arbre*/
			evaluate(&child,&child_result,my_rank,status,NP);

			/* Le résultat du dernier coup (ie la dernière case du tableau) correspond au résultat de l'appel récursif de la ligne précédente*/
			tableau[n_moves-1]=child_result;

			/*On prévient les processus esclaves que l'on a fini la distribution des coups de la profondeur*/
			for(i=0;i<NP-1;i++){
				continuer=0;	
				MPI_Recv(&a, 1,MPI_INT ,MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, status);
				MPI_Send(&continuer, 1, MPI_INT, status->MPI_SOURCE, TAG, MPI_COMM_WORLD);
			}

			/*0 calcule le résultat*/
			for(i=0;i<n_moves;i++){
				if(i!=n_moves-1){
					MPI_Wait(request+i, status);
				}
				if(tableau[i].score!=620004059){ //si le child a évalué un coup
					int child_score = -tableau[i].score;

				if (child_score > result->score) {	
					result->score = child_score;
					result->best_move = moves[i];
					result->pv_length = tableau[i].pv_length + 1;
					int j;
					for(j = 0; j < tableau[i].pv_length; j++)	
						result->PV[j+1] = tableau[i].PV[j];
					result->PV[0] = moves[i];
					}	
				}
			}	

		}

		/*Profondeur comprise entre 0 et P (exclus)*/
		else if(T->height!=0 && T->height < P){

			/* Création de la structure arbre pour MPI */
			const int nbChampsArbre=14;
			int tailleChampsArbre[14] = {128,128,1,1,1,1,1,1,2,2,128,1,1, MAX_DEPTH};
			MPI_Datatype TypesChampsArbre[14] = {MPI_CHAR, MPI_CHAR, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_CHAR, MPI_INT, MPI_INT, MPI_INT};
			MPI_Datatype MPI_TREE_T;
			MPI_Aint StructArbre[14];

			StructArbre[0] = offsetof(tree_t, pieces);
			StructArbre[1] = offsetof(tree_t, colors);
			StructArbre[2] = offsetof(tree_t, side);
			StructArbre[3] = offsetof(tree_t, depth);
			StructArbre[4] = offsetof(tree_t, height);
			StructArbre[5] = offsetof(tree_t, alpha);
			StructArbre[6] = offsetof(tree_t, beta);
			StructArbre[7] = offsetof(tree_t, alpha_start);
			StructArbre[8] = offsetof(tree_t, king);
			StructArbre[9] = offsetof(tree_t, pawns);
			StructArbre[10] = offsetof(tree_t, attack);
			StructArbre[11] = offsetof(tree_t, suggested_move);
			StructArbre[12] = offsetof(tree_t, hash);
			StructArbre[13] = offsetof(tree_t, history);

			MPI_Type_create_struct(nbChampsArbre, tailleChampsArbre, StructArbre, TypesChampsArbre, &MPI_TREE_T);
			MPI_Type_commit(&MPI_TREE_T);

			/*Distribution des n_moves-1 coups*/
			for (i = 0; i < n_moves-1; i++) {
				continuer=3;
				MPI_Recv(&a, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, status);
				MPI_Send(&continuer, 1, MPI_INT, status->MPI_SOURCE, TAG, MPI_COMM_WORLD);
				MPI_Send(T,1,MPI_TREE_T,status->MPI_SOURCE, TAG, MPI_COMM_WORLD);
				MPI_Send(&(moves[i]), 1, MPI_INT, status->MPI_SOURCE, TAG, MPI_COMM_WORLD);
				MPI_Irecv(tableau+i,1,MPI_RESULT_T,status->MPI_SOURCE,TAG,MPI_COMM_WORLD,request+i);
			}
			tree_t child;

			/*On joue le dernier coup*/
			play_move(T, moves[n_moves-1], &child);

			/*Le processus 0 descend une nouvelle fois récursivement à droite de l'arbre*/
			evaluate(&child,&child_result,my_rank,status,NP);

			/* Le résultat du dernier coup (ie la dernière case du tableau) correspond au résultat de l'appel récursif de la ligne précédente*/
			tableau[n_moves-1]=child_result;

			/*0 calcule le résultat*/
			for(i=0;i<n_moves;i++){
				if(i!=n_moves-1){
					MPI_Wait(request+i, status);
				}
				if(tableau[i].score!=620004059){ //si le child a évalué un coup
					int child_score = -tableau[i].score;

					if (child_score > result->score) {	
						result->score = child_score;
						result->best_move = moves[i];
						result->pv_length = tableau[i].pv_length + 1;
						int j;

						for(j = 0; j < tableau[i].pv_length; j++){	
							result->PV[j+1] = tableau[i].PV[j];
						}
						result->PV[0] = moves[i];
					}	
				}
			}

		}

		/*Si la profondeur est >=P, 0 ne descend plus récursivement à droite de l'arbre*/
		else if(T->height >= P){

		/* Création de la structure arbre pour MPI */
			const int nbChampsArbre=14;
			int tailleChampsArbre[14] = {128,128,1,1,1,1,1,1,2,2,128,1,1, MAX_DEPTH};
			MPI_Datatype TypesChampsArbre[14] = {MPI_CHAR, MPI_CHAR, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_CHAR, MPI_INT, MPI_INT, MPI_INT};
			MPI_Datatype MPI_TREE_T;
			MPI_Aint StructArbre[14];

			StructArbre[0] = offsetof(tree_t, pieces);
			StructArbre[1] = offsetof(tree_t, colors);
			StructArbre[2] = offsetof(tree_t, side);
			StructArbre[3] = offsetof(tree_t, depth);
			StructArbre[4] = offsetof(tree_t, height);
			StructArbre[5] = offsetof(tree_t, alpha);
			StructArbre[6] = offsetof(tree_t, beta);
			StructArbre[7] = offsetof(tree_t, alpha_start);
			StructArbre[8] = offsetof(tree_t, king);
			StructArbre[9] = offsetof(tree_t, pawns);
			StructArbre[10] = offsetof(tree_t, attack);
			StructArbre[11] = offsetof(tree_t, suggested_move);
			StructArbre[12] = offsetof(tree_t, hash);
			StructArbre[13] = offsetof(tree_t, history);

			MPI_Type_create_struct(nbChampsArbre, tailleChampsArbre, StructArbre, TypesChampsArbre, &MPI_TREE_T);
			MPI_Type_commit(&MPI_TREE_T);

			/*Distribution de tous les coups*/
			for (i = 0; i < n_moves; i++) {
				continuer=3;
				MPI_Recv(&a, 1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, status);
				MPI_Send(&continuer, 1, MPI_INT, status->MPI_SOURCE, TAG, MPI_COMM_WORLD);
				MPI_Send(T,1,MPI_TREE_T,status->MPI_SOURCE, TAG, MPI_COMM_WORLD);
				MPI_Send(&(moves[i]), 1, MPI_INT, status->MPI_SOURCE, TAG, MPI_COMM_WORLD);
				MPI_Irecv(tableau+i,1,MPI_RESULT_T,status->MPI_SOURCE,TAG,MPI_COMM_WORLD,request+i);
			}

			/*0 calcule le résultat*/
			for(i=0;i<n_moves;i++){
				if(i!=n_moves-1){
					MPI_Wait(request+i, status);
				}
				if(tableau[i].score!=620004059){ //si le child a évalué un coup
					int child_score = -tableau[i].score;

					if (child_score > result->score) {	
						result->score = child_score;
						result->best_move = moves[i];
						result->pv_length = tableau[i].pv_length + 1;
						int j;

						for(j = 0; j < tableau[i].pv_length; j++){	
							result->PV[j+1] = tableau[i].PV[j];
						}

						result->PV[0] = moves[i];
					}	
				}
			}
		}
	
		free(tableau);
		free(request);
	}

	/*Les autres processus appelent evaluate en séquentiel*/
	else {
		int i;
		/*on parallélise avec des threads les appels récursifs des processus esclaves : on distribue aux threads les coups issus de la seconde profondeur */
		if(T->height==2){
		#pragma omp parallel for schedule(runtime)
			for (i = 0; i < n_moves; i++) {
				tree_t child;
				result_t child_result;

				play_move(T, moves[i], &child);

				evaluate(&child, &child_result,my_rank,status,NP);

				int child_score = -child_result.score;

				if (child_score > result->score) {
					result->score = child_score;

					result->best_move = moves[i];
					result->pv_length = child_result.pv_length + 1;
					int j;
					for(j = 0; j < child_result.pv_length; j++)
						result->PV[j+1] = child_result.PV[j];
					result->PV[0] = moves[i];
				}

				T->alpha = MAX(T->alpha, child_score);
			}
				if (TRANSPOSITION_TABLE)
          				tt_store(T, result);
			}

			else{
				for (i = 0; i < n_moves; i++) {
					tree_t child;
					result_t child_result;

					play_move(T, moves[i], &child);

					evaluate(&child, &child_result,my_rank,status,NP);

					int child_score = -child_result.score;

					if (child_score > result->score) {
						result->score = child_score;

						result->best_move = moves[i];
						result->pv_length = child_result.pv_length + 1;
						int j;
						for(j = 0; j < child_result.pv_length; j++)
							result->PV[j+1] = child_result.PV[j];
						result->PV[0] = moves[i];
					}



					if (ALPHA_BETA_PRUNING && child_score >= T->beta)
					break;    

					T->alpha = MAX(T->alpha, child_score);
				}
				if (TRANSPOSITION_TABLE)
          				tt_store(T, result);

			}	

        }

}


void evaluateSeq(tree_t * T, result_t *result)
{
        node_searched++;
  	
        move_t moves[MAX_MOVES];
        int n_moves;

        result->score = -MAX_SCORE - 1;
        result->pv_length = 0;
        
        if (test_draw_or_victory(T, result))
          return;

        if (TRANSPOSITION_TABLE && tt_lookup(T, result))     /* la réponse est-elle déjà connue ? */
          return;
        
        compute_attack_squares(T);

        /* profondeur max atteinte ? si oui, évaluation heuristique */
        if (T->depth == 0) {
          result->score = (2 * T->side - 1) * heuristic_evaluation(T);
          return;
        }
        
        n_moves = generate_legal_moves(T, &moves[0]);

        /* absence de coups légaux : pat ou mat */
	if (n_moves == 0) {
          result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
          return;
        }
        
        if (ALPHA_BETA_PRUNING)
          sort_moves(T, n_moves, moves);

        /* évalue récursivement les positions accessibles à partir d'ici */
		int i;
		
		/*Si l'on est à la racine de l'arbre, on distribue les coups à différents threads*/
		if(T->height==0){
			#pragma omp parallel for schedule(runtime)
		    for (i = 0; i < n_moves; i++) {
			tree_t child;
		        result_t child_result;
		            
		        play_move(T, moves[i], &child);
	        
		        evaluateSeq(&child, &child_result)
;             
		        int child_score = -child_result.score;

			if (child_score > result->score) {
				result->score = child_score;
				result->best_move = moves[i];
				int j;
				result->pv_length = child_result.pv_length + 1;
				for(j = 0; j < child_result.pv_length; j++)
					result->PV[j+1] = child_result.PV[j];
				result->PV[0] = moves[i];
	           	 }


			T->alpha = MAX(T->alpha, child_score);
		    }
		}

		else{
			for (i = 0; i < n_moves; i++) {
				tree_t child;
				result_t child_result;

				play_move(T, moves[i], &child);


				evaluateSeq(&child, &child_result);             
				int child_score = -child_result.score;


				if (child_score > result->score) {
					result->score = child_score;
					result->best_move = moves[i];
					int j;
					result->pv_length = child_result.pv_length + 1;

					for(j = 0; j < child_result.pv_length; j++)
						result->PV[j+1] = child_result.PV[j];
					result->PV[0] = moves[i];
				}

				T->alpha = MAX(T->alpha, child_score);
			}
		}
}


void decide(tree_t * T, result_t *result, int NP, int my_rank, MPI_Status* status)
{
	int depth, continuer,i;
	for (depth = 1;; depth++) {
		T->depth = depth;
		T->height = 0;
		T->alpha_start = T->alpha = -MAX_SCORE - 1;
		T->beta = MAX_SCORE + 1;
		
		printf("=====================================\n");

		/*on parallélise à partir d'une profondeur palier*/
		if(depth>PROF_PARAL){
			evaluate(T, result,my_rank,status,NP);
		}
		else{			
			evaluateSeq(T,result);
		}
		
        	printf("depth: %d / score: %.2f / best_move : ", T->depth, 0.01 * result->score);

        	print_pv(T, result);

		/*on vient ici si on a le résultat définitif*/
       		 if (DEFINITIVE(result->score)){
			if(depth>PROF_PARAL){
		  		continuer=0;
		  		for(i=1;i<NP;i++){
					MPI_Send(&continuer, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
		  		}
			}

			/*Si on a tout fait en séquentiel, on dit aux processus esclaves de s'arrêter*/
			else{
				int cpt;
				for(cpt=0;cpt<NP-1;cpt++){
					int a,continuer=0;
					MPI_Recv(&a,1,MPI_INT,MPI_ANY_SOURCE,TAG,MPI_COMM_WORLD,status);
					MPI_Send(&continuer,1,MPI_INT,status->MPI_SOURCE,TAG,MPI_COMM_WORLD);
					MPI_Send(&continuer,1,MPI_INT,status->MPI_SOURCE,TAG,MPI_COMM_WORLD);
				}
			}
        	break;
		}

		else{
			if(depth>PROF_PARAL){
				continuer=1;
				 for(i=1;i<NP;i++){
					MPI_Send(&continuer, 1, MPI_INT, i, TAG, MPI_COMM_WORLD);
				 }
			}	
		}	
	}
}

int main(int argc, char **argv)
{  
	/*Initialisation de la connexion*/
	int NP;
	int my_rank;
	int provided;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
	MPI_Comm_size(MPI_COMM_WORLD, &NP);
  	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  	MPI_Status status;
  
	double debut=0.0, fin=0.0;
	tree_t root;
	result_t result;

	/*vérification des paramètres*/
	if(my_rank==0){
		if (argc < 2) {
			printf("usage: %s \"4k//4K/4P w\" (or any position in FEN)\n", argv[0]);
			exit(1);
		}
	}

	if (ALPHA_BETA_PRUNING)
		printf("Alpha-beta pruning ENABLED\n");

	if (TRANSPOSITION_TABLE) {
		printf("Transposition table ENABLED\n");
		init_tt();
	}
		    
	parse_FEN(argv[1], &root);

	if(my_rank==0){
		print_position(&root);
	
		debut = my_gettimeofday();    
		decide(&root, &result,NP,my_rank,&status);
		
	}

	else{
			
		/* Création de la structure arbre pour MPI */
		const int nbChampsArbre=14;
		int          tailleChampsArbre[14] = {128,128,1,1,1,1,1,1,2,2,128,1,1, MAX_DEPTH};
		MPI_Datatype TypesChampsArbre[14] = {MPI_CHAR, MPI_CHAR, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_CHAR, MPI_INT, MPI_INT, MPI_INT};
		MPI_Datatype MPI_TREE_T;
		MPI_Aint     StructArbre[14];

		StructArbre[0] = offsetof(tree_t, pieces);
		StructArbre[1] = offsetof(tree_t, colors);
		StructArbre[2] = offsetof(tree_t, side);
		StructArbre[3] = offsetof(tree_t, depth);
		StructArbre[4] = offsetof(tree_t, height);
		StructArbre[5] = offsetof(tree_t, alpha);
		StructArbre[6] = offsetof(tree_t, beta);
		StructArbre[7] = offsetof(tree_t, alpha_start);
		StructArbre[8] = offsetof(tree_t, king);
		StructArbre[9] = offsetof(tree_t, pawns);
		StructArbre[10] = offsetof(tree_t, attack);
		StructArbre[11] = offsetof(tree_t, suggested_move);
		StructArbre[12] = offsetof(tree_t, hash);
		StructArbre[13] = offsetof(tree_t, history);

		MPI_Type_create_struct(nbChampsArbre, tailleChampsArbre, StructArbre, TypesChampsArbre, &MPI_TREE_T);
		MPI_Type_commit(&MPI_TREE_T);	  			

		/* Création de la structure result pour MPI */
		const int nbChampsResult=4;
		int tailleChampsResult[4] = {1,1,1, MAX_DEPTH};
		MPI_Datatype TypesChampsResult[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
		MPI_Datatype MPI_RESULT_T;
		MPI_Aint     StructResult[4];

		StructResult[0] = offsetof(result_t, score);
		StructResult[1] = offsetof(result_t, best_move);
		StructResult[2] = offsetof(result_t, pv_length);
		StructResult[3] = offsetof(result_t, PV);

		MPI_Type_create_struct(nbChampsResult, tailleChampsResult, StructResult, TypesChampsResult, &MPI_RESULT_T);
		MPI_Type_commit(&MPI_RESULT_T);


		root.height = 1;
		root.alpha_start = root.alpha = -MAX_SCORE - 1;
		root.beta = MAX_SCORE + 1;

		result.score=620004059; //score par défaut, si ça marche, le mettre dans le #define

		/*Les processus esclaves récupèrent les fils que 0 leur transmet et calculent les résultats de ceux-ci*/
		while(1){
			tree_t child;
			tree_t arbre_recu;
			int move;
			int continuer; //1 pour continuer, 0 sinon
			MPI_Send(&my_rank, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD);
			MPI_Recv(&continuer, 1,MPI_INT ,0, TAG, MPI_COMM_WORLD, &status);
			if(continuer==1){
				MPI_Recv(&(root.depth), 1,MPI_INT ,0, TAG, MPI_COMM_WORLD, &status);
				MPI_Recv(&move, 1,MPI_INT ,0, TAG, MPI_COMM_WORLD, &status);
				play_move(&root, move, &child);
				evaluate(&child,&result,my_rank,&status,NP);
				MPI_Send(&result,1,MPI_RESULT_T,0, TAG, MPI_COMM_WORLD);
			}

			else if(continuer==0){
				MPI_Recv(&continuer, 1,MPI_INT ,0, TAG, MPI_COMM_WORLD, &status);
				if(continuer==0){
					break;
				}		
			}
			
			/* si on est à une profondeur >0 */
			else if(continuer==3){
				MPI_Recv(&(arbre_recu), 1,MPI_TREE_T ,0, TAG, MPI_COMM_WORLD, &status);
				MPI_Recv(&move, 1,MPI_INT ,0, TAG, MPI_COMM_WORLD, &status);
				play_move(&arbre_recu, move, &child);
				evaluate(&child,&result,my_rank,&status,NP);
				MPI_Send(&result,1,MPI_RESULT_T,0, TAG, MPI_COMM_WORLD);
			}
		}
	}

	/*Affichage du résultat*/
	if(my_rank==0){
		fin = my_gettimeofday();
	  	fprintf( stdout, "total computation time (with gettimeofday()) : %g s\n",
		   fin - debut);
		printf("\nDécision de la position: ");
		switch(result.score * (2*root.side - 1)) {
			case MAX_SCORE: printf("blanc gagne\n"); 
				break;
			case CERTAIN_DRAW: printf("partie nulle\n"); 
				break;
			case -MAX_SCORE: printf("noir gagne\n"); 
				break;
			default: 
				printf("BUG\n");
		}
	}
        //printf("Node searched: %llu\n", node_searched);
        
        if (TRANSPOSITION_TABLE)
          free_tt();

	MPI_Finalize();
	return 0;
}
