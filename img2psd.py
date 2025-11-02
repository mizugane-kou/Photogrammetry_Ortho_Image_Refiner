# pip install opencv-python pytoshop numpy Pillow pykakasi six

import os
import sys
import cv2
import pytoshop
from pytoshop import layers
import numpy as np
from PIL import Image
from datetime import datetime
import pykakasi

def get_max_image_size(image_paths):
    """
    指定された画像パスのリストから、最大の幅と高さを取得する。
    PillowのDecompression Bomb対策としてcv2を使用。
    """
    heights = []
    widths = []

    for img_path in image_paths:
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"画像のデコードに失敗しました（cv2）: {img_path}")
                continue
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
        except Exception as e:
            print(f"画像サイズの取得中にエラーが発生しました: {img_path}\n詳細: {e}")
            continue

    W = max(widths) if widths else None
    H = max(heights) if heights else None
    return W, H

def convert_to_romaji(text):
    """
    日本語の文字列をローマ字（ヘボン式）に変換する。
    """
    kakasi = pykakasi.kakasi()
    kakasi.setMode("H", "a")  # ひらがなをローマ字に変換
    kakasi.setMode("K", "a")  # カタカナをローマ字に変換
    kakasi.setMode("J", "a")  # 漢字をローマ字に変換
    kakasi.setMode("r", "Hepburn")  # ヘボン式ローマ字
    converter = kakasi.getConverter()
    return converter.do(text)

def main(image_paths):
    """
    複数の画像ファイルを読み込み、1つのPSDファイルにレイヤーとしてまとめる。
    各レイヤーの位置関係は元画像のキャンバスサイズに基づくが、
    バウンディングボックスは描画範囲に一致させる。
    """
    if not image_paths:
        print("画像ファイルが指定されていません。")
        return

    # PSDファイル全体のキャンバスサイズを決定
    W, H = get_max_image_size(image_paths)
    if not W or not H:
        print("有効な画像ファイルが見つかりません。")
        return

    # PSDオブジェクト作成（RGBA想定）
    psd = pytoshop.core.PsdFile(num_channels=4, height=H, width=W)

    # 各画像をレイヤーとして追加
    for img_path in image_paths:
        try:
            img_array = np.fromfile(img_path, np.uint8)
            test_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"画像の読み込み中にエラー: {img_path}\n詳細: {e}")
            continue

        if test_img is None:
            print(f"画像の読み込みに失敗しました: {img_path}")
            continue

        # アルファチャンネルを分離
        if test_img.shape[2] == 4:
            rgba = test_img
            rgb = rgba[:, :, :3]
            alpha = rgba[:, :, 3]
        else:
            rgb = test_img
            alpha = np.full(test_img.shape[:2], 255, dtype=np.uint8)

        # --- 描画範囲（非透過部分）のバウンディングボックスを算出 ---
        coords = cv2.findNonZero((alpha > 0).astype(np.uint8))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            rgb_cropped = rgb[y:y+h, x:x+w]
            alpha_cropped = alpha[y:y+h, x:x+w]
        else:
            print(f"完全透明のためスキップ: {img_path}")
            continue

        # --- トリミング後のピクセルデータをチャンネルごとに分解 ---
        layer_1 = layers.ChannelImageData(image=alpha_cropped, compression=1)
        layer0 = layers.ChannelImageData(image=rgb_cropped[:, :, 2], compression=1)
        layer1 = layers.ChannelImageData(image=rgb_cropped[:, :, 1], compression=1)
        layer2 = layers.ChannelImageData(image=rgb_cropped[:, :, 0], compression=1)

        # ファイル名（拡張子なし）をレイヤー名に
        layer_name = os.path.splitext(os.path.basename(img_path))[0]
        layer_name_romaji = convert_to_romaji(layer_name)

        # --- レイヤーを作成 ---
        new_layer = layers.LayerRecord(
            channels={-1: layer_1, 0: layer0, 1: layer1, 2: layer2},
            top=y, bottom=y + h, left=x, right=x + w,  # 元の位置を維持しつつバウンディングボックスを狭める
            name=layer_name_romaji,
            opacity=255,
        )

        # PSDに追加
        psd.layer_and_mask_info.layer_info.layer_records.append(new_layer)

    # 出力ファイル名生成
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"output_{timestamp}.psd"

    try:
        with open(output_filename, 'wb') as fd2:
            psd.write(fd2)
        print(f"PSDファイルを保存しました: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"PSDファイルの保存中にエラーが発生しました。\n詳細: {e}")

if __name__ == '__main__':
    image_paths = sys.argv[1:]
    main(image_paths)
