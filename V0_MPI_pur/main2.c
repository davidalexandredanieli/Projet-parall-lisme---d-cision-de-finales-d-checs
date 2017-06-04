#include "projet.h"
#include <mpi.h>
#include <sys/time.h>
/* 2017-02-23 : version 0.0 MPI pur */

double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}
unsigned long long int node_searched = 0;

void evaluate(tree_t * T, result_t *result,int my_rank, int* continuer, int tag, MPI_Status status, int* modulo)
{

	/* on rentre dans ce if si 1 nous a prévenu qu'un autre processus a fini (on peut revenir souvent ici à cause du "for récursif") */
	if(*continuer==0){
		return;
	}

	/*Clef de notre parallélisation, nous avons ajusté ce modulo pour éviter que les communications soient trop fréquentes*/
	if(*modulo%50000==0){
		/*on envoie notre valeur de continuer à 1 et on reçoit une valeur en retour qui nous dit s'il faut continuer ou non*/
		MPI_Send(continuer,1,MPI_INT,1,1,MPI_COMM_WORLD);
		MPI_Recv(continuer,1,MPI_INT,1,1,MPI_COMM_WORLD,&status);
	}

	(*modulo)++;

	/* on rentre dans ce if si 1 nous a prévenu qu'un autre processus a fini */
	if(*continuer==0){
		return;
	}
	
        node_searched++; //compte la profondeur
        move_t moves[MAX_MOVES];
        int n_moves;
        result->score = -MAX_SCORE - 1;
        result->pv_length = 0; //taille du chemin dans l'arbre
        
        if (test_draw_or_victory(T, result)){ //détermine si la partie est nulle ou gagnée
			return;
		}
      	        
        compute_attack_squares(T); //détermine les cases attaquées par chaque camp

        /* profondeur max atteinte ? si oui, évaluation heuristique */
        if (T->depth == 0) {
			result->score = (2 * T->side - 1) * heuristic_evaluation(T);
			return;
        }
        
        n_moves = generate_legal_moves(T, &moves[0]); //on écrit dans le tableau move les coups légaux

        /* absence de coups légaux : pat ou mat */
		if (n_moves == 0) {
			result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
			return;
		}

        /* évalue récursivement les positions accessibles à partir d'ici */
        int i;
        for ( i = 0; i < n_moves; i++) {
			tree_t child;
           	result_t child_result;
            play_move(T, moves[i], &child); //écrit dans child la position obtenue en jouant move   
            evaluate(&child, &child_result,my_rank,continuer, tag, status,modulo); //fonction récursive avec les nouvelles positions                         
            int child_score = -child_result.score;

			if (child_score > result->score) { //result score vaut autre chose que -Maxscore-1 si la prof max est atteinte ou on ne peut plus joueur
			result->score = child_score;
			result->best_move = moves[i];
            result->pv_length = child_result.pv_length + 1;
        	int j;
            for(j = 0; j < child_result.pv_length; j++)
                result->PV[j+1] = child_result.PV[j]; //on met dans les noeuds parents du meilleur coup la valeur de ce coup
            result->PV[0] = moves[i];
            }
        }
}


void decide(tree_t * T, result_t *result,int my_rank,int p)
{
	MPI_Status status;
	int modulo=0; //sert à limiter les communications entre le processus 1 et les processus qui calculent*/
	int source;
	int dest;
	int tag=0;
	int continuer=1; //vaut 1 si l'on doit continuer, 0 sinon
	int depth=0;
	int depth_current;
	int depth_tmp;
	int cpt=0;
	int temp;
	int continuerTemp=1;

	if(my_rank!=0 && my_rank!=1){
		while(1){
			/*on envoie notre valeur de continuer à 1 et on reçoit une valeur en retour qui nous dit s'il faut continuer ou non*/
			MPI_Send(&continuer,1,MPI_INT,1,1,MPI_COMM_WORLD);
			MPI_Recv(&continuer,1,MPI_INT,1,1,MPI_COMM_WORLD,&status);
						
			/*on envoie au processus 0 si l'on a fini ou non : continuer=1->pas fini, continuer=0->fini*/
			MPI_Send(&continuer,1,MPI_INT,0,1,MPI_COMM_WORLD);

			if(continuer==0){ 				
				break;
			}

			/*on reçoit la réponse de 0, il est possible qu'un autre processus ait fini et que 0 nous dise d'arrêter*/
			MPI_Recv(&continuer,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);

			/*si 0 dit au processus d'arrêter, il s'arrête*/
			if(continuer==0){ 
				break;
			}
			/*sinon on continue et 0 envoie au processus la profondeur à explorer*/
			MPI_Recv(&depth,1,MPI_INT,0,4,MPI_COMM_WORLD,&status);

			/*on met à jour l'arbre*/
			T->depth=depth;
			T->height=0;
			T->alpha_start=T->alpha=-MAX_SCORE-1;
			T->beta=MAX_SCORE+1;

			/*on fait appel à evaluate en transmettant un pointeur sur la variable continuer pour qu'elle soit éventuellement modifiée*/
			evaluate(T,result,my_rank,&continuer,tag,status,&modulo);

			/*on rentre si le processus 1 lui a dit qu'un autre processus a trouvé le résultat de la partie*/
			if(continuer==0){
				printf("Fin du processus %d\n",my_rank);
				/*on prévient 0 que le processus va s'arrêter*/
				MPI_Send(&continuer,1,MPI_INT,0,1,MPI_COMM_WORLD);
				break;
			}
			
			/*on rentre dans ce if si le processus a trouvé le résultat de la partie*/
			if(DEFINITIVE(result->score)){ 
				continuer=0;
				/*le processus prévient 0 et 1 qu'il a terminé*/
				MPI_Send(&continuer,1,MPI_INT,0,1,MPI_COMM_WORLD);
				MPI_Send(&continuer,1,MPI_INT,1,1,MPI_COMM_WORLD);
				/*on met ça pour identifier le processus qui connait le résultat de la partie (champ rajouté dans la structure)*/
				result->processus_gagnant=1;
				printf("Fin du processus %d\n",my_rank);
				break;	
			}
		}
	}


	if(my_rank==0){
			/*for qui ditribue à chaque processus les profondeurs à explorer*/
			for (depth_current = 1;; depth_current++) {
				/*tant que tous les processus n'ont pas terminé*/
				if(cpt<p-2){
					/*le processus dit à 0 s'il a terminé*/
					MPI_Recv(&continuer,1,MPI_INT,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&status);
					
					/*si cpt!=0 c'est que le résultat a déjà été trouvé, on introduit un compteur qui compte le nombre de processus qui s'arrêtent*/
					if(cpt!=0){cpt++;}
					
					/*si le résultat n'avait pas encore été trouvé*/
					else{ 
						/*soit le processus a fini et on incrémente le compteur...*/
						MPI_Send(&continuer,1,MPI_INT,status.MPI_SOURCE,1,MPI_COMM_WORLD);
						if(continuer==0){
							cpt++;
						}
						/*... soit on redonne du travail au processus*/			
						else{
							MPI_Send(&depth_current,1,MPI_INT,status.MPI_SOURCE,4,MPI_COMM_WORLD);
						}
					}
				}

				/*si tous les processus ont terminé 0 termine*/
				else{
					printf("Fin du processus %d\n",my_rank);
					break;				
				}
			}
	}

	if(my_rank==1){
		
		while(cpt<p-2){
			/*le processus 1 reçoit des autres processus des indications pour savoir s'ils ont fini, si c'est la cas...*/
			MPI_Recv(&continuerTemp,1,MPI_INT,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&status);
			if(continuerTemp==0){
				/*...il incrémente le compteur qui compte les processus finis et met continuer à 0*/
				continuer=0;
				cpt++;
			}
			else{
				/*sinon il prévient le processus de s'il faut continuer ou non*/
				continuer=continuer*continuerTemp;
				MPI_Send(&continuer,1,MPI_INT,status.MPI_SOURCE,1,MPI_COMM_WORLD);
				/* on rentre dans ce if quand on doit prévenir le processus que c'est terminé*/
				if(continuer!=1){
					cpt++;
				}
			}
		}
		printf("Fin du processus %d\n", my_rank);
	}
		
                 
}


int main(int argc, char **argv)
{  
	double debut=0.0, fin=0.0;
	int my_rank;
	int p;
	unsigned long long int node_searched_global;
	tree_t root;
	result_t result;

    if (argc < 2) {
      printf("usage: %s \"4k//4K/4P w\" (or any position in FEN)\n", argv[0]);
      exit(1);
    }

       
    parse_FEN(argv[1], &root);
	
	
	MPI_Init(&argc, &argv);
	debut = my_gettimeofday(); 	
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);


	result.processus_gagnant=0;
	if(my_rank==0){
		print_position(&root);
    }
        
	decide(&root, &result,my_rank,p);
	fin = my_gettimeofday();
  	fprintf( stdout, "total computation time (with gettimeofday()) : %g s\n",fin - debut);
	/*sert à compter le nombre de noeuds parcourus*/
	MPI_Reduce(&node_searched, &node_searched_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,MPI_COMM_WORLD);
	
	if(my_rank==0){
		printf("Total nodes searched: %llu\n",node_searched_global);
	}
	/*tous les processus s'attendent pour éviter des problèmes d'affichage*/
	MPI_Barrier(MPI_COMM_WORLD);
	/*le processus ayant trouvé le résultat va l'afficher*/
	if(result.processus_gagnant==1){
		printf("===================================\n");
		printf("depth: %d / score: %.2f / best_move : ", root.depth, 0.01 * result.score);
		print_pv(&root, &result);			
		
		printf("\nDécision de la position: ");
		switch(result.score * (2*root.side - 1)) {
			case MAX_SCORE: 
				printf("blanc gagne\n"); 
			break;
			case CERTAIN_DRAW: 
				printf("partie nulle\n"); 
			break;
			case -MAX_SCORE: 
				printf("noir gagne\n"); 
			break;
			default: 
				printf("BUG\n");
		}
		printf("\n===================================\n");

	}
	

	MPI_Finalize();
	return 0;
}
