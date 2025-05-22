import os
import sys
import codecs
import json
import logging
import traceback
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    current_app,
    send_from_directory
)
from werkzeug.utils import secure_filename
from .models import (
    KNNModelis,
    CNNModelis,
    TransformerModelis,
    KNNModelDB,
    CNNModelDB,
    TransformerModelDB,
    ModelResults,
    TestResult,
    init_db
)
from app.models.database import db
from .utils import (
    process_image,
    augment_image,
    prepare_data,
    save_data,
    load_data,
    load_dataset,
    save_dataset_info,
    load_dataset_info,
    sukurti_konfuzijos_macica,
    sukurti_klasiu_grafikus,
    sukurti_metriku_grafikus,
    issaugoti_rezultatus
)
from app.utils.zenklu_zodynas import ZENKLU_ZODYNAS
from app.utils.dataset_loader import nuskaityti_is_csv, nuskaityti_is_zip
from app.utils.data_processing import universalus_duomenu_nuskaitymas, load_dataset_cnn

from app.model_progress import model_progress

# Nustatyti UTF-8 koduotę
# sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
# sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Sukurti Blueprint
bp = Blueprint('main', __name__)

# Sukurti modelius
knn_model = KNNModelis()
cnn_model = CNNModelis()
transformer_model = TransformerModelis()

# Pagrindinis puslapis
@bp.route('/')
def index():
    # Surinkti modelių sąrašus kaip ir rezultatų puslapyje
    knn_modeliai, cnn_modeliai, transformer_modeliai = [], [], []
    models_dir = 'app/static/models'
    if os.path.exists(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith('.joblib') or fname.endswith('.joblib.joblib'):
                if fname.startswith('knn_'):
                    knn_modeliai.append({'pavadinimas': fname})
                elif fname.startswith('cnn_'):
                    cnn_modeliai.append({'pavadinimas': fname})
                elif fname.startswith('transformer_'):
                    transformer_modeliai.append({'pavadinimas': fname})
    return render_template('index.html',
                          knn_modeliai=knn_modeliai,
                          cnn_modeliai=cnn_modeliai,
                          transformer_modeliai=transformer_modeliai)

# Mokymo puslapis
@bp.route('/mokymas')
def mokyti():
    return render_template('mokymas.html')

# Testavimo puslapis
@bp.route('/testuoti', methods=['GET', 'POST'])
def testuoti():
    if request.method == 'POST':
        try:
            # Gauti duomenis iš formos
            image = request.files['image']
            model_type = request.form['model_type']
            
            # Išsaugoti paveikslėlį
            filename = secure_filename(image.filename)
            image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            
            # Apdoroti paveikslėlį
            processed_image = process_image(image_path)
            
            # Prognozuoti
            if model_type == 'knn':
                model = knn_model
            elif model_type == 'cnn':
                model = cnn_model
            else:
                model = transformer_model
                
            prediction = model.prognozuoti(processed_image)
            
            return jsonify({
                'success': True,
                'prediction': int(prediction),
                'confidence': float(model.confidence)
            })
            
        except Exception as e:
            logging.error(f"Klaida testuojant modelį: {str(e)}")
            return jsonify({'success': False, 'message': str(e)})
            
    return render_template('testuoti.html')

# Rezultatų puslapis
@bp.route('/rezultatai')
def rezultatai():
    import glob
    try:
        # Mokymo rezultatai
        results = []
        naudotos_klases = set()
        results_model_names = set()
        # Surinkti visus JSON failus iš app/results/, app/results/knn/, app/results/cnn/, app/results/transformer/
        search_dirs = [
            os.path.join('app', 'results'),
            os.path.join('app', 'results', 'knn'),
            os.path.join('app', 'results', 'cnn'),
            os.path.join('app', 'results', 'transformer')
        ]
        for d in search_dirs:
            if os.path.exists(d):
                for filename in os.listdir(d):
                    if filename.endswith('.json'):
                        with open(os.path.join(d, filename), 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            # Remap metrikos fields to correct Lithuanian keys
                            if 'rezultatai' in result:
                                metrikos = result['rezultatai']
                                # Map possible unicode-encoded keys to correct ones
                                key_map = {
                                    'preciziškumas': ['preciziškumas', 'precizi\u0161kumas'],
                                    'atgaminimas': ['atgaminimas'],
                                    'tikslumas': ['tikslumas'],
                                    'f1_balas': ['f1_balas', 'f1\u005fbalas'],
                                }
                                for correct, variants in key_map.items():
                                    for v in variants:
                                        if v in metrikos and correct != v:
                                            metrikos[correct] = metrikos.pop(v)
                                # If metrikos is a subdict, map English keys to Lithuanian
                                if 'metrikos' in metrikos and isinstance(metrikos['metrikos'], dict):
                                    eng2lt = {
                                        'accuracy': 'tikslumas',
                                        'precision': 'preciziškumas',
                                        'recall': 'atgaminimas',
                                        'f1': 'f1_balas',
                                    }
                                    for eng, lt in eng2lt.items():
                                        if eng in metrikos['metrikos']:
                                            metrikos[lt] = metrikos['metrikos'][eng]
                                    # Optionally remove the subdict
                                    del metrikos['metrikos']
                            results.append(result)
                            # Surenkame modelių pavadinimus iš rezultatų
                            if 'modelio_pavadinimas' in result:
                                results_model_names.add(result['modelio_pavadinimas'])
                            elif 'modelio_tipas' in result and 'data' in result:
                                results_model_names.add(f"{result['modelio_tipas']}_{result['data']}")
                            if 'naudotos_klases' in result:
                                naudotos_klases.update(result['naudotos_klases'])
        if not naudotos_klases and results:
            for res in results:
                if 'rezultatai' in res and 'naudotos_klases' in res['rezultatai']:
                    naudotos_klases.update(res['rezultatai']['naudotos_klases'])
        naudotos_klases = sorted(list(naudotos_klases))
        naudotu_zenklu_pavadinimai = [ZENKLU_ZODYNAS.get(int(k), str(k)) for k in naudotos_klases]
        results.sort(key=lambda x: x['data'], reverse=True)

        # Surinkime visus modelių failus
        knn_modeliai = []
        cnn_modeliai = []
        transformer_modeliai = []
        models_dir = 'app/static/models'
        for fname in os.listdir(models_dir):
            if fname.endswith('.joblib') or fname.endswith('.joblib.joblib'):
                if fname.startswith('knn_'):
                    knn_modeliai.append(fname)
                elif fname.startswith('cnn_'):
                    cnn_modeliai.append(fname)
                elif fname.startswith('transformer_'):
                    transformer_modeliai.append(fname)

        # Filtruoti results pagal modelio tipą, jei pasirinktas
        modelio_tipas = request.args.get('modelio_tipas')
        filtered_results = results
        if modelio_tipas:
            filtered_results = [r for r in results if r.get('modelio_tipas') == modelio_tipas]

        # Modelių sąrašus paversti į žodynų sąrašus su 'pavadinimas'
        knn_modeliai_dict = [{'pavadinimas': fname} for fname in knn_modeliai]
        cnn_modeliai_dict = [{'pavadinimas': fname} for fname in cnn_modeliai]
        transformer_modeliai_dict = [{'pavadinimas': fname} for fname in transformer_modeliai]

        # Testavimo rezultatai
        query = TestResult.query
        if modelio_tipas:
            query = query.filter_by(modelio_tipas=modelio_tipas)
        test_results = query.order_by(TestResult.data.desc()).all()

        return render_template('rezultatai.html', 
                             knn_results=next((r for r in results if r.get('modelio_tipas') == 'knn'), None),
                             cnn_results=next((r for r in results if r.get('modelio_tipas') == 'cnn'), None),
                             transformer_results=next((r for r in results if r.get('modelio_tipas') == 'transformer'), None),
                             knn_modeliai=knn_modeliai_dict,
                             cnn_modeliai=cnn_modeliai_dict,
                             transformer_modeliai=transformer_modeliai_dict,
                             naudotu_zenklu_pavadinimai=naudotu_zenklu_pavadinimai,
                             test_results=test_results,
                             results=filtered_results
        )
    except FileNotFoundError as e:
        return render_template('rezultatai.html', 
                             knn_results=None,
                             cnn_results=None,
                             transformer_results=None,
                             knn_modeliai=[],
                             cnn_modeliai=[],
                             transformer_modeliai=[],
                             naudotu_zenklu_pavadinimai=[], 
                             test_results=[],
                             klaida=f'Failas nerastas: {str(e)}')
    except Exception as e:
        return render_template('rezultatai.html', 
                             knn_results=None,
                             cnn_results=None,
                             transformer_results=None,
                             knn_modeliai=[],
                             cnn_modeliai=[],
                             transformer_modeliai=[],
                             naudotu_zenklu_pavadinimai=[], 
                             test_results=[],
                             klaida=f'Įvyko nenumatyta klaida: {str(e)}')

# Progreso puslapis
@bp.route('/progresas')
def progresas():
    return jsonify(model_progress.get_all())

# Statinių failų puslapis
@bp.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(current_app.config['STATIC_FOLDER'], filename)

@bp.route('/grafikai')
def grafikai():
    modelio_tipas = request.args.get('modelis', 'knn')
    katalogai = {
        'knn': 'knn_grafikai',
        'cnn': 'cnn_grafikai',
        'transformer': 'transformer_grafikai'
    }
    katalogas = katalogai.get(modelio_tipas, 'knn_grafikai')
    pilnas_katalogas = os.path.join('app/static', katalogas)
    if not os.path.exists(pilnas_katalogas):
        grafikai = []
    else:
        grafikai = [f for f in os.listdir(pilnas_katalogas) if f.endswith('.png')]
    grafikai = [f'{katalogas}/{f}' for f in grafikai]
    return render_template('grafikai.html', grafikai=grafikai, modelio_tipas=modelio_tipas)

@bp.route('/json-failai')
def json_failai():
    failai = [f for f in os.listdir('app/static') if f.endswith('.json')]
    return render_template('json_failai.html', failai=failai)

@bp.route('/perziureti_json_faila/<failo_pavadinimas>')
def perziureti_json_faila(failo_pavadinimas):
    try:
        with open(os.path.join('app/static', failo_pavadinimas), 'r', encoding='utf-8') as f:
            turinys = json.load(f)
        return render_template('json_perziura.html', failo_pavadinimas=failo_pavadinimas, turinys=turinys)
    except Exception as e:
        return render_template('json_perziura.html', klaida=str(e))

@bp.route('/importuoti_rezultatus')
def importuoti_rezultatus():
    try:
        import json
        from app.models.knn_model import KNNModelis
        import os
        failas = 'app/static/knn_porciju_rezultatai.json'
        if not os.path.exists(failas):
            return render_template('importuoti_rezultatus.html', message='Rezultatų failas nerastas.'), 404
        with open(failas, encoding='utf-8') as f:
            rezultatai = json.load(f)
        modelis = KNNModelis()
        kiek = 0
        with current_app.app_context():
            for res in rezultatai:
                try:
                    metrikos = res.get('kitos_metrikos', {})
                    metrikos['tikslumas'] = res.get('tikslumas', 0.0)
                    metrikos['preciziškumas'] = res.get('preciziskumas', 0.0)
                    metrikos['atgaminimas'] = res.get('atgaminimas', 0.0)
                    metrikos['f1_balas'] = res.get('f1_balas', 0.0)
                    modelis.issaugoti_rezultatus(metrikos)
                    kiek += 1
                except Exception as e:
                    current_app.logger.error(f"Klaida įrašant rezultatą: {str(e)}")
                    continue
        return render_template('importuoti_rezultatus.html', message=f'Į DB importuota rezultatų: {kiek}')
    except Exception as e:
        current_app.logger.error(f"Klaida importuojant rezultatus: {str(e)}")
        return render_template('importuoti_rezultatus.html', message=f'Įvyko klaida: {str(e)}'), 500 

@bp.route('/informacija')
def informacija():
    return render_template('informacija.html')

@bp.route('/naudojimas')
def naudojimas():
    return render_template('naudojimas.html')

@bp.route('/prognoze', methods=['POST'])
def prognoze():
    try:
        if 'file' not in request.files:
            return jsonify({'klaida': 'Nepateiktas failas'}), 400
        failas = request.files['file']
        modelio_tipas = request.form.get('modelio_tipas', 'knn')
        # Get the correct modelio_pavadinimas field depending on modelio_tipas
        if modelio_tipas == 'knn':
            modelio_pavadinimas = request.form.get('modelio_pavadinimas_knn')
        elif modelio_tipas == 'cnn':
            modelio_pavadinimas = request.form.get('modelio_pavadinimas_cnn')
        elif modelio_tipas == 'transformer':
            modelio_pavadinimas = request.form.get('modelio_pavadinimas_transformer')
        else:
            modelio_pavadinimas = None
        if failas.filename == '':
            return jsonify({'klaida': 'Nepasirinktas failas'}), 400

        # Išsaugome paveiksliuką
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        paveiksliuko_kelias = os.path.join('app/static/test_images', f'test_{timestamp}_{failas.filename}')
        os.makedirs(os.path.dirname(paveiksliuko_kelias), exist_ok=True)
        failas.save(paveiksliuko_kelias)

        apdorotas_paveikslėlis = process_image(paveiksliuko_kelias)
        # KNN atveju: grayscale, 32x32, flatten
        if modelio_tipas == 'knn':
            import cv2
            import numpy as np
            if apdorotas_paveikslėlis.shape[-1] == 3:
                apdorotas_paveikslėlis = cv2.cvtColor((apdorotas_paveikslėlis * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                apdorotas_paveikslėlis = (apdorotas_paveikslėlis * 255).astype(np.uint8)
            apdorotas_paveikslėlis = cv2.resize(apdorotas_paveikslėlis, (32, 32))
            apdorotas_paveikslėlis = apdorotas_paveikslėlis.flatten().reshape(1, -1) / 255.0
        # CNN/transformer atveju paliekame kaip yra
        
        # Gauname modelio failo pavadinimą
        if modelio_tipas == 'knn' and modelio_pavadinimas:
            if modelio_pavadinimas.endswith('.joblib'):
                modelio_failas = f'app/static/models/{modelio_pavadinimas}'
            else:
                modelio_failas = f'app/static/models/knn_{modelio_pavadinimas}.joblib'
        elif modelio_tipas == 'knn':
            modelio_failas = 'app/static/models/knn_model.joblib'
        elif modelio_tipas == 'cnn' and modelio_pavadinimas:
            if modelio_pavadinimas.endswith('.joblib'):
                modelio_failas = f'app/static/models/{modelio_pavadinimas}'
            else:
                modelio_failas = f'app/static/models/cnn_{modelio_pavadinimas}.joblib'
        elif modelio_tipas == 'cnn':
            modelio_failas = 'app/static/models/cnn_model.joblib'
        elif modelio_tipas == 'transformer' and modelio_pavadinimas:
            if modelio_pavadinimas.endswith('.joblib'):
                modelio_failas = f'app/static/models/{modelio_pavadinimas}'
            else:
                modelio_failas = f'app/static/models/transformer_{modelio_pavadinimas}.joblib'
        elif modelio_tipas == 'transformer':
            modelio_failas = 'app/static/models/transformer_model.joblib'
        else:
            return jsonify({'klaida': 'Neteisingas modelio tipas'}), 400

        # Užkrauname modelį
        if modelio_tipas == 'knn':
            from app.models.knn_model import KNNModelis
            knn_modelis = KNNModelis()
            if os.path.exists(modelio_failas):
                import joblib
                knn_modelis.model = joblib.load(modelio_failas)
            else:
                return jsonify({'klaida': 'Modelis nerastas'}), 400
            prognozė, tikimybė = knn_modelis.prognozuoti(apdorotas_paveikslėlis)
        elif modelio_tipas == 'cnn':
            modelio_failas = 'app/static/models/cnn_model.joblib'
            if cnn_model.modelis is None:
                if os.path.exists(modelio_failas):
                    cnn_model.ikelti_modeli(modelio_failas)
                else:
                    try:
                        cnn_model.ikelti_is_db()
                    except Exception:
                        return jsonify({'klaida': 'Modelis dar nebuvo išmokytas ir nėra išsaugotas.'}), 400
            prognozė, tikimybė = cnn_model.prognozuoti(apdorotas_paveikslėlis)
        elif modelio_tipas == 'transformer':
            modelio_failas = 'app/static/models/transformer_model.joblib'
            if transformer_model.modelis is None:
                if os.path.exists(modelio_failas):
                    transformer_model.ikelti_modeli(modelio_failas)
                else:
                    try:
                        transformer_model.ikelti_is_db()
                    except Exception:
                        return jsonify({'klaida': 'Modelis dar nebuvo išmokytas ir nėra išsaugotas.'}), 400
            prognozė, tikimybė = transformer_model.prognozuoti(apdorotas_paveikslėlis)
        else:
            return jsonify({'klaida': 'Neteisingas modelio tipas'}), 400

        # Išsaugome rezultatą duomenų bazėje
        zenklo_pavadinimas = ZENKLU_ZODYNAS.get(int(prognozė), str(prognozė))
        test_result = TestResult(
            modelio_tipas=modelio_tipas,
            modelio_pavadinimas=modelio_pavadinimas,
            atpazintas_zenklas=int(prognozė),
            zenklo_pavadinimas=zenklo_pavadinimas,
            paveiksliuko_kelias=paveiksliuko_kelias,
            tikimybe=float(tikimybė)
        )
        db.session.add(test_result)
        db.session.commit()

        original_img_path = None
        if prognozė and prognozė is not None:
            # 1. Ieškome static/meta/
            static_meta_dir = os.path.join('app', 'static', 'meta')
            for ext in ['png', 'jpg', 'jpeg']:
                candidate = os.path.join(static_meta_dir, f"{prognozė}.{ext}")
                if os.path.exists(candidate):
                    original_img_path = f"meta/{prognozė}.{ext}"
                    break
            # 2. Jei nerado static/meta/, bandome duomenys/Meta/ (debugui)
            if not original_img_path:
                meta_dir = os.path.join('duomenys', 'Meta')
                for ext in ['png', 'jpg', 'jpeg']:
                    candidate = os.path.join(meta_dir, f"{prognozė}.{ext}")
                    if os.path.exists(candidate):
                        original_img_path = candidate
                        break

        return jsonify({
            'prognozė': prognozė,
            'zenklo_pavadinimas': zenklo_pavadinimas,
            'tikimybė': tikimybė,
            'paveiksliuko_kelias': paveiksliuko_kelias,
            'original_img_path': original_img_path
        })
    except FileNotFoundError as e:
        return jsonify({'klaida': f'Failas nerastas: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'klaida': f'Neteisingi duomenys: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'klaida': f'Įvyko nenumatyta klaida: {str(e)}'}), 500 

@bp.route('/train', methods=['POST'])
def train():
    try:
        print('request.form:', dict(request.form))
        print('request.files:', dict(request.files))
        model_type = request.form.get('modelio_tipas', request.form.get('model_type', 'cnn'))
        modelio_pavadinimas = request.form.get('modelio_pavadinimas', 'modelis')
        augment = request.form.get('augment', 'false').lower() == 'true'

        # Tikrinti ar įkeltas failas
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = secure_filename(file.filename)
            save_path = os.path.join('duomenys', filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)
            csv_path = save_path
            data_dir = 'duomenys'  # katalogas, kuriame yra Train/...
        else:
            # Naudoti numatytąjį failą
            csv_path = 'data/raw/annotations.csv'
            data_dir = 'data/raw'

        print('Kviečiu load_dataset su:', data_dir, csv_path)
        if model_type == 'cnn':
            print('Naudoju load_dataset_cnn CNN modeliui')
            X, y = load_dataset_cnn(data_dir, csv_path)
        else:
            X, y = load_dataset(data_dir, csv_path)
        print('Po load_dataset:', type(X), getattr(X, 'shape', None), type(y), getattr(y, 'shape', None))
        print('Po load_dataset: X example:', X[:1] if hasattr(X, '__getitem__') else X)
        print('Po load_dataset: y example:', y[:5] if hasattr(y, '__getitem__') else y)

        # Paruošti duomenis
        X_processed, y_processed = prepare_data(X, y, augment=augment, model_type=model_type)
        print('Po prepare_data: X_processed type:', type(X_processed), 'X_processed shape:', getattr(X_processed, 'shape', None))
        print('Po prepare_data: y_processed type:', type(y_processed), 'y_processed shape:', getattr(y_processed, 'shape', None))
        print('Po prepare_data: y_processed example:', y_processed[:5] if hasattr(y_processed, '__getitem__') else y_processed)

        # Mokyti modelį
        if model_type == 'knn':
            rezultatai = knn_model.mokyti(X_processed, y_processed)
        elif model_type == 'cnn':
            rezultatai = cnn_model.mokyti(X_processed, y_processed)
        else:
            rezultatai = transformer_model.mokyti(X_processed, y_processed)

        return jsonify({'success': True, 'results': rezultatai})

    except Exception as e:
        logging.error(f'Klaida mokant modelį: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

@bp.route('/stop_training', methods=['POST'])
def stop_training():
    try:
        model_type = request.form.get('model_type')
        if not model_type or model_type not in ['knn', 'cnn', 'transformer']:
            return jsonify({'klaida': 'Neteisingas modelio tipas'}), 400
            
        model_progress.stop(model_type)
        return jsonify({'message': f'Stopping {model_type} training...'})
    except Exception as e:
        return jsonify({'klaida': str(e)}), 500 

@bp.route('/prognoze_rezultatas')
def prognoze_rezultatas():
    from app.models.database import TestResult
    import os
    prognoze = TestResult.query.order_by(TestResult.data.desc()).first()
    original_img_path = None
    if prognoze and prognoze.atpazintas_zenklas is not None:
        # 1. Ieškome static/meta/
        static_meta_dir = os.path.join('app', 'static', 'meta')
        for ext in ['png', 'jpg', 'jpeg']:
            candidate = os.path.join(static_meta_dir, f"{prognoze.atpazintas_zenklas}.{ext}")
            if os.path.exists(candidate):
                original_img_path = f"meta/{prognoze.atpazintas_zenklas}.{ext}"
                break
        # 2. Jei nerado static/meta/, bandome duomenys/Meta/ (debugui)
        if not original_img_path:
            meta_dir = os.path.join('duomenys', 'Meta')
            for ext in ['png', 'jpg', 'jpeg']:
                candidate = os.path.join(meta_dir, f"{prognoze.atpazintas_zenklas}.{ext}")
                if os.path.exists(candidate):
                    original_img_path = candidate
                    break
    return render_template('prognoze.html', prognoze=prognoze, original_img_path=original_img_path) 